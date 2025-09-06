#!/usr/bin/env python3
"""
Анализатор JSONL файлов с помощью OpenRouter API
"""

import json
import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import requests

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('jsonl_analyzer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
# Исправляем проблему с путями для WSL/Windows
try:
    # Сначала пробуем найти .env в текущей директории
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        logger.info(f"Загружен .env файл из: {env_path}")
    else:
        # Если не найден, пробуем стандартный поиск
        load_dotenv()
        logger.info("Загружен .env файл через стандартный поиск")
except Exception as e:
    logger.warning(f"Не удалось загрузить .env файл: {e}")
    # Продолжаем работу, переменные окружения могут быть установлены системно

class OpenRouterClient:
    """Клиент для работы с OpenRouter API"""
    
    def __init__(self):
        self.api_key = os.getenv('OPENROUTER_API_KEY')
        logger.info(f"API ключ из переменных окружения: {'найден' if self.api_key else 'не найден'}")
        
        if not self.api_key:
            # Пробуем загрузить из .env файла напрямую
            try:
                # Пробуем разные пути к .env файлу
                possible_paths = [
                    Path(__file__).parent / '.env',
                    Path('.env'),
                    Path('/home/andrey/projects/Cursor/2_JSONL_AI/.env'),
                    Path('//wsl.localhost/Ubuntu-24.04/home/andrey/projects/Cursor/2_JSONL_AI/.env')
                ]
                
                for env_path in possible_paths:
                    logger.info(f"Проверяем путь: {env_path}")
                    if env_path.exists():
                        logger.info(f"Найден .env файл: {env_path}")
                        with open(env_path, 'r', encoding='utf-8') as f:
                            for line in f:
                                if line.startswith('OPENROUTER_API_KEY='):
                                    self.api_key = line.split('=', 1)[1].strip()
                                    logger.info("API ключ загружен из .env файла")
                                    break
                        if self.api_key:
                            break
                    else:
                        logger.info(f"Файл не найден: {env_path}")
                        
            except Exception as e:
                logger.warning(f"Не удалось прочитать .env файл: {e}")
        
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY не найден в переменных окружения или .env файле")
        
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "JSONL Analyzer"
        }
        
        # Список моделей в порядке приоритета
        self.models = [
            "meta-llama/llama-3.1-8b-instruct:free",
            "microsoft/phi-3-mini-128k-instruct:free",
            "google/gemini-flash-1.5:free",
            "meta-llama/llama-3.1-70b-instruct",
            "anthropic/claude-3.5-sonnet"
        ]
        self.current_model_index = 0
        
        # Лимиты API
        self.rate_limit_delay = 1.0  # секунд между запросами
        self.max_retries = 3
        self.timeout = 30
    
    def get_current_model(self) -> str:
        """Получить текущую модель"""
        return self.models[self.current_model_index]
    
    def switch_to_next_model(self) -> bool:
        """Переключиться на следующую модель"""
        if self.current_model_index < len(self.models) - 1:
            self.current_model_index += 1
            logger.info(f"Переключение на модель: {self.get_current_model()}")
            return True
        return False
    
    def generate_title(self, post_text: str) -> Optional[str]:
        """Генерация заголовка для поста"""
        from prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
        
        payload = {
            "model": self.get_current_model(),
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(post_text=post_text)}
            ],
            "max_tokens": 100,
            "temperature": 0.3,
            "top_p": 0.9
        }
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Попытка {attempt + 1} с моделью {self.get_current_model()}")
                
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    data = response.json()
                    title = data['choices'][0]['message']['content'].strip()
                    logger.info(f"Заголовок сгенерирован: {title[:50]}...")
                    return title
                
                elif response.status_code == 429:  # Rate limit
                    wait_time = self.rate_limit_delay * (2 ** attempt)
                    logger.warning(f"Rate limit. Ожидание {wait_time} секунд...")
                    time.sleep(wait_time)
                    continue
                
                elif response.status_code in [400, 401, 403]:
                    logger.error(f"Ошибка API: {response.status_code} - {response.text}")
                    if self.switch_to_next_model():
                        continue
                    else:
                        break
                
                else:
                    logger.error(f"Неожиданная ошибка: {response.status_code} - {response.text}")
                    if self.switch_to_next_model():
                        continue
                    else:
                        break
                        
            except requests.exceptions.Timeout:
                logger.warning(f"Таймаут запроса. Попытка {attempt + 1}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.rate_limit_delay)
                    continue
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Ошибка запроса: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.rate_limit_delay)
                    continue
            
            # Задержка между попытками
            if attempt < self.max_retries - 1:
                time.sleep(self.rate_limit_delay)
        
        logger.error(f"Не удалось сгенерировать заголовок для поста")
        return None

class JSONLAnalyzer:
    """Анализатор JSONL файлов"""
    
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.client = OpenRouterClient()
        
        # Статистика
        self.processed = 0
        self.failed = 0
        self.retry_queue = []
    
    def process_jsonl_file(self, file_path: Path) -> None:
        """Обработка одного JSONL файла"""
        logger.info(f"Обработка файла: {file_path}")
        
        output_file = self.output_dir / f"analyzed_{file_path.name}"
        
        with open(file_path, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            
            for line_num, line in enumerate(infile, 1):
                try:
                    data = json.loads(line.strip())
                    
                    # Извлекаем текст поста (адаптируй под структуру твоего JSONL)
                    post_text = self.extract_post_text(data)
                    
                    if not post_text:
                        logger.warning(f"Строка {line_num}: пустой текст поста")
                        continue
                    
                    # Генерируем заголовок
                    title = self.client.generate_title(post_text)
                    
                    if title:
                        # Добавляем заголовок в начало поста
                        data['title'] = title
                        data['original_text'] = post_text
                        data['processed_text'] = f"{title}\n\n{post_text}"
                        
                        outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                        self.processed += 1
                        logger.info(f"Строка {line_num}: обработана успешно")
                    else:
                        # Добавляем в очередь для повторной обработки
                        self.retry_queue.append((line_num, data, post_text))
                        self.failed += 1
                        logger.warning(f"Строка {line_num}: не удалось сгенерировать заголовок")
                    
                    # Задержка между запросами
                    time.sleep(self.client.rate_limit_delay)
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Строка {line_num}: ошибка JSON - {e}")
                    continue
                except Exception as e:
                    logger.error(f"Строка {line_num}: неожиданная ошибка - {e}")
                    continue
    
    def extract_post_text(self, data: Dict[str, Any]) -> str:
        """Извлечение текста поста из JSON объекта"""
        # Адаптируй под структуру твоего JSONL
        # Возможные поля для текста:
        text_fields = ['текст поста', 'text', 'content', 'post', 'message', 'body']
        
        for field in text_fields:
            if field in data and data[field]:
                return str(data[field]).strip()
        
        # Если не найден стандартный текст, попробуем все строковые поля
        for key, value in data.items():
            if isinstance(value, str) and len(value) > 50:  # Увеличил минимальную длину
                return value.strip()
        
        return ""
    
    def retry_failed_items(self) -> None:
        """Повторная обработка неудачных элементов"""
        if not self.retry_queue:
            return
        
        logger.info(f"Повторная обработка {len(self.retry_queue)} элементов...")
        
        # Сбрасываем индекс модели для повторной попытки
        self.client.current_model_index = 0
        
        retry_output_file = self.output_dir / "retry_analyzed.jsonl"
        
        with open(retry_output_file, 'w', encoding='utf-8') as outfile:
            for line_num, data, post_text in self.retry_queue:
                title = self.client.generate_title(post_text)
                
                if title:
                    data['title'] = title
                    data['original_text'] = post_text
                    data['processed_text'] = f"{title}\n\n{post_text}"
                    
                    outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                    self.processed += 1
                    self.failed -= 1
                    logger.info(f"Повторная обработка строки {line_num}: успешно")
                else:
                    logger.error(f"Повторная обработка строки {line_num}: снова неудача")
                
                time.sleep(self.client.rate_limit_delay)
    
    def run(self) -> None:
        """Запуск анализа"""
        logger.info("Запуск анализа JSONL файлов")
        
        # Поиск JSONL файлов в входной директории
        jsonl_files = list(self.input_dir.glob("*.jsonl"))
        
        if not jsonl_files:
            logger.error(f"JSONL файлы не найдены в {self.input_dir}")
            return
        
        logger.info(f"Найдено {len(jsonl_files)} JSONL файлов")
        
        for file_path in jsonl_files:
            self.process_jsonl_file(file_path)
        
        # Повторная обработка неудачных элементов
        if self.retry_queue:
            self.retry_failed_items()
        
        # Итоговая статистика
        logger.info(f"Анализ завершен:")
        logger.info(f"  Обработано успешно: {self.processed}")
        logger.info(f"  Не удалось обработать: {self.failed}")
        logger.info(f"  Результаты сохранены в: {self.output_dir}")

def main():
    """Главная функция"""
    input_dir = "/home/andrey/projects/Cursor/1_CSV_1/input"  # WSL путь
    output_dir = "/home/andrey/projects/Cursor/2_JSONL_AI/output"
    
    try:
        analyzer = JSONLAnalyzer(input_dir, output_dir)
        analyzer.run()
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        raise

if __name__ == "__main__":
    main()