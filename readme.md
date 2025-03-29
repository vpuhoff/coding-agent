# Техническое Задание (ТЗ)

**Название проекта:** Автономный AI-агент для разработки и рефакторинга кода ("CodeCraft AI")

**Версия:** 1.0

**Дата:** 26.05.2024

---

## 1. Общие положения

### 1.1. Назначение
Разработка программного агента (далее – `Агент`), способного автономно выполнять задачи по добавлению новой функциональности или рефакторингу в существующие кодовые базы веб-проектов (`React`, `Vue`, `Angular`, `Node.js`, `Python`/`Django`/`Flask` и др.) на основе текстового описания задачи. Результатом работы `Агента` является Pull Request (`PR`) в системе контроля версий (например, `GitHub`, `GitLab`), готовый к ревью человеком.

### 1.2. Цели
- Автоматизировать рутинные задачи разработки и модификации кода.
- Ускорить процесс внесения изменений в кодовую базу.
- Снизить когнитивную нагрузку на разработчиков при работе с незнакомым кодом.
- Обеспечить генерацию кода, соответствующего стилю и паттернам проекта.
- Гарантировать работоспособность изменений путем автоматического запуска тестов и статического анализа.

### 1.3. Заказчик
(Внутренняя разработка / Конкретный отдел)

### 1.4. Исполнитель
(Команда разработки)

### 1.5. Термины и определения
- **`Агент`**: Программный комплекс, выполняющий задачи данного ТЗ.
- **`LLM` (Large Language Model)**: Большая языковая модель (например, Google Gemini API), используемая для анализа, генерации и понимания текста и кода.
- **`Задача`**: Текстовое описание функциональности, которую нужно реализовать или изменить.
- **`Проект`**: Репозиторий с исходным кодом, над которым работает `Агент`.
- **`Песочница` (Sandbox)**: Изолированное окружение (например, `Docker`-контейнер) для безопасного клонирования, сборки, запуска, тестирования и анализа кода `Проекта`.
- **`Артефакты анализа`**: Результаты работы статических анализаторов (линтеры, тайпчекеры) и систем тестирования.
- **`Патч`**: Набор изменений в коде, сгенерированный `Агентом`.
- **`PR` (Pull Request)**: Запрос на слияние изменений, созданный `Агентом` в системе контроля версий.

---

## 2. Требования к функциональности

### 2.1. Прием и анализ задачи
- `Агент` должен принимать на вход текстовое описание `Задачи` и `URL` репозитория `Проекта` (с указанием ветки, от которой создавать новую).
- `Агент` должен использовать `LLM` для парсинга и структурирования `Задачи`: выявления ключевых требований, затронутых областей кода (UI, API, DB), необходимых изменений.
- `Агент` должен иметь возможность запросить уточнения, если `Задача` неясна (механизм уточнения `TBD` – возможно, через комментарий к тикету или специальный интерфейс).

### 2.2. Подготовка окружения
- `Агент` должен клонировать указанный репозиторий `Проекта` в `Песочницу`.
- `Агент` должен создать новую ветку для `Задачи` от указанной базовой ветки.
- `Агент` должен автоматически определять основные технологии `Проекта` (язык, фреймворк, менеджер пакетов - `npm`/`yarn`/`pip`) на основе файлов конфигурации (`package.json`, `requirements.txt`, `pom.xml` и т.д.) и анализа кода.
- `Агент` должен пытаться автоматически определить команды для установки зависимостей, сборки, запуска тестов и статического анализа из файлов конфигурации (например, `scripts` в `package.json`). Если определить команды автоматически не удалось, должна быть возможность указать их в конфигурации `Агента` для данного `Проекта`.
- `Агент` должен устанавливать все необходимые зависимости `Проекта` внутри `Песочницы`.

### 2.3. Анализ кодовой базы
- `Агент` должен выполнить анализ структуры `Проекта` (основные каталоги, их назначение).
- Используя `LLM` и анализ кода (`AST`-деревья, поиск по файлам), `Агент` должен:
    - Определить потенциальные точки входа – файлы/модули/классы/компоненты, которые наиболее вероятно потребуют изменений для выполнения `Задачи`.
    - Найти связанные участки кода (вызовы функций, использование компонентов, взаимодействие со стейтом, API-вызовы).
    - Идентифицировать основные паттерны кодирования, используемые в релевантных частях проекта (стиль, обработка ошибок, управление состоянием, запросы к API, тестирование).

### 2.4. Генерация и применение изменений (Итеративный процесс)

#### 2.4.1. Генерация Патча
На основе анализа `Задачи` и кодовой базы, `Агент` должен использовать `LLM` для генерации конкретных изменений (`Патч`):
- Создание новых файлов.
- Модификация существующих файлов (в формате, пригодном для автоматического применения, например, `diff` или инструкции по замене блоков).

#### 2.4.2. Применение Патча
`Агент` должен безопасно применить сгенерированный `Патч` к коду `Проекта` внутри `Песочницы`.

#### 2.4.3. Проверка Сборки/Компиляции
Если применимо для языка/фреймворка, `Агент` должен запустить команду сборки/компиляции и проверить ее успешность.

#### 2.4.4. Запуск Статического Анализа
`Агент` должен запустить настроенные в `Проекте` статические анализаторы (линтеры, тайпчекеры).

#### 2.4.5. Запуск Тестов
`Агент` должен запустить все тесты (юнит, интеграционные), определенные в `Проекте`.

#### 2.4.6. Анализ Результатов и Отладка
- `Агент` должен парсить результаты сборки, статического анализа и тестов.
- При обнаружении ошибок:
    - `Агент` должен передать информацию об ошибках (логи, сообщения) и соответствующий код в `LLM`.
    - `LLM` должен попытаться проанализировать ошибку и предложить исправление (новый `Патч`).
    - `Агент` должен применить исправление и повторить шаги 2.4.2-2.4.6.
- Должен быть установлен лимит на количество попыток отладки для предотвращения зацикливания.

#### 2.4.7. Повторение
Если после применения `Патча` все проверки (сборка, анализ, тесты) прошли успешно, `Агент` переходит к следующему шагу. Если нет и лимит отладки не исчерпан, возвращается к 2.4.6. Если лимит исчерпан, `Агент` должен сообщить о невозможности выполнить `Задачу` и предоставить логи.

### 2.5. Генерация Тестов
- После успешного применения изменений для основной функциональности, `Агент` должен сгенерировать новые юнит-тесты (и, возможно, интеграционные), покрывающие добавленную/измененную функциональность, следуя существующим паттернам тестирования в `Проекте`.
- `Агент` должен добавить сгенерированные тесты в кодовую базу и повторно запустить все тесты (шаг 2.4.5), включая новые, и при необходимости выполнить отладку (шаг 2.4.6).

### 2.6. Коммит и создание Pull Request
- `Агент` должен проиндексировать все изменения (новый/измененный код, тесты).
- `Агент` должен сгенерировать осмысленное сообщение коммита (возможно, с помощью `LLM` на основе `Задачи`).
- `Агент` должен выполнить коммит в созданную ранее ветку.
- `Агент` должен отправить (`push`) ветку в удаленный репозиторий.
- `Агент` должен создать `Pull Request` в системе контроля версий (например, через `GitHub`/`GitLab` `API`):
    - Заголовок `PR` должен быть основан на `Задаче`.
    - Описание `PR` должно включать:
        - Краткое изложение `Задачи`.
        - Список внесенных изменений (ключевые файлы, общая суть).
        - Указание на то, что тесты и статический анализ пройдены успешно (или отчет об ошибках, если отладка не удалась).
        - Призыв к ревью человеком.

### 2.7. Отчетность
- `Агент` должен предоставить пользователю ссылку на созданный `PR`.
- `Агент` должен сохранять логи своей работы для возможного анализа проблем.

---

## 3. Архитектурные требования

### 3.1. Модульность
`Агент` должен иметь модульную архитектуру, где каждый основной этап (анализ задачи, подготовка окружения, анализ кода, генерация, проверка, отладка, создание `PR`) реализован как отдельный, слабо связанный компонент.

### 3.2. Песочница (Sandboxing)
Все операции с кодом `Проекта` (клонирование, установка зависимостей, сборка, запуск, тестирование, анализ) должны выполняться в изолированном, воспроизводимом окружении (рекомендовано использование `Docker`). Это обеспечивает безопасность и предотвращает влияние на основную систему.

### 3.3. Управление состоянием
Должен быть реализован механизм управления состоянием `Агента` на протяжении выполнения `Задачи` (хранение промежуточных результатов анализа, сгенерированного кода, статуса проверок, логов).

### 3.4. Конфигурация
Ключевые параметры `Агента` (`API`-ключи для `LLM` и `VCS`, лимиты на попытки отладки, возможно, специфичные команды для проектов) должны быть вынесены в конфигурационные файлы или переменные окружения.

### 3.5. Абстракция LLM
Взаимодействие с `LLM` должно быть вынесено в отдельный слой абстракции, чтобы потенциально можно было заменить модель (например, `Gemini` на `GPT` или другую) с минимальными изменениями в основной логике `Агента`.

### 3.6. Обработка ошибок и отказоустойчивость
`Агент` должен корректно обрабатывать ошибки на каждом этапе (сетевые проблемы, ошибки парсинга, ошибки выполнения команд, ошибки `API`) и предоставлять информативные сообщения об ошибках. Должны быть предусмотрены механизмы повторных попыток для временных сбоев.

---

## 4. Требования к компонентам

### 4.1. Оркестратор (Workflow Engine)
- Управляет общим потоком выполнения `Задачи`.
- Вызывает другие модули в нужной последовательности.
- Управляет состоянием `Агента`.
- Реализует логику итераций и отладочного цикла.

### 4.2. Менеджер Песочницы (Sandbox Manager)
- Отвечает за создание, настройку и удаление изолированных окружений (`Docker`-контейнеров).
- Предоставляет интерфейс для выполнения команд внутри `Песочницы`.
- Управляет файловой системой внутри `Песочницы` (клонирование репо, применение патчей).

### 4.3. Сервис LLM (LLM Service Wrapper)
- Инкапсулирует взаимодействие с `API` `LLM`.
- Формирует промпты для различных задач (анализ, генерация, отладка).
- Обрабатывает и парсит ответы `LLM` (включая `JSON`).
- Управляет контекстом диалога с `LLM`.

### 4.4. Анализатор Кода (Code Analyzer)
- Использует инструменты статического анализа (парсинг `AST`, поиск зависимостей) и `Сервис LLM` для анализа структуры, паттернов и связей в коде.
- Определяет команды сборки/тестирования/анализа.

### 4.5. Генератор Кода (Code Generator)
- Использует `Сервис LLM` для генерации `Патчей` и сообщений коммитов/`PR`.

### 4.6. Исполнитель Проверок (Verification Runner)
- Запускает команды сборки, статического анализа и тестов внутри `Песочницы` через `Менеджер Песочницы`.
- Парсит вывод команд для определения статуса (успех/неудача) и извлечения сообщений об ошибках.

### 4.7. Отладчик (Debugger Assistant)
- Получает информацию об ошибках от `Исполнителя Проверок`.
- Взаимодействует с `Сервисом LLM` для получения предложений по исправлению.
- Формирует `Патчи` с исправлениями.

### 4.8. Менеджер VCS (Version Control Manager)
- Взаимодействует с `Git` (создание веток, коммиты, `push`) внутри `Песочницы`.
- Взаимодействует с `API` хостинга репозиториев (`GitHub`/`GitLab`) для создания `PR`.

---

## 5. Нефункциональные требования

### 5.1. Безопасность
Обеспечить изоляцию выполнения кода `Проекта` через `Песочницу`. `API`-ключи и другие секреты должны храниться безопасно.

### 5.2. Производительность
Время выполнения `Задачи` должно быть приемлемым (зависит от сложности `Задачи` и размера `Проекта`, но не должно занимать многие часы для типичных задач). Оптимизировать взаимодействие с `LLM` и выполнение команд.

### 5.3. Надежность
`Агент` должен быть устойчив к сбоям и корректно обрабатывать ошибки.

### 5.4. Конфигурируемость
Возможность настройки параметров `Агента` и специфичных команд для разных проектов.

### 5.5. Логирование
Ведение подробных логов работы `Агента` для диагностики проблем.

---

## 6. План разработки (Примерные этапы)

### Этап 1 (MVP Core)
- Реализация `Оркестратора`, `Менеджера Песочницы`, базового `Сервиса LLM`, `Менеджера VCS` (только клонирование и работа с ветками).
- Реализация приема `Задачи` и подготовки окружения.
- Реализация базового `Анализатора Кода` (определение технологий, поиск файлов).
- **Цель:** `Агент` может клонировать репо, создать ветку, определить технологии.

### Этап 2 (Code Analysis & Generation)
- Улучшение `Анализатора Кода` (поиск точек входа, паттернов с помощью `LLM`).
- Реализация `Генератора Кода` (генерация `Патчей`).
- Реализация применения `Патчей` в `Песочнице`.
- **Цель:** `Агент` может проанализировать код и сгенерировать/применить изменения по `Задаче` (без проверок).

### Этап 3 (Verification)
- Реализация `Исполнителя Проверок` (запуск сборки, линтеров, тестов).
- Парсинг результатов проверок.
- **Цель:** `Агент` может генерировать, применять и проверять изменения статическим анализом и тестами.

### Этап 4 (Debugging Loop & Testing)
- Реализация `Отладчика` (анализ ошибок, генерация исправлений через `LLM`).
- Интеграция отладочного цикла в `Оркестратор`.
- Реализация генерации тестов для новой функциональности.
- **Цель:** `Агент` может итеративно исправлять ошибки в сгенерированном коде и генерировать тесты.

### Этап 5 (PR Creation & Finalization)
- Реализация коммита изменений.
- Интеграция с `API` `GitHub`/`GitLab` для создания `PR`.
- Реализация отчетности и логирования.
- **Цель:** Полнофункциональный `Агент`, создающий готовый к ревью `PR`.

### Этап 6 (Refinement & Optimization)
- Оптимизация производительности.
- Улучшение обработки ошибок.
- Расширение поддержки фреймворков/языков.
- Добавление пользовательского интерфейса (опционально).

---

## 7. Критерии приемки
- `Агент` успешно выполняет >X% тестовых `Задач` разной сложности на Y различных репозиториях-примерах.
- `Агент` корректно работает с указанными системами контроля версий (`GitHub`/`GitLab`).
- Все проверки (сборка, статический анализ, тесты) в `Песочнице` проходят для успешно созданных `PR`.
- Код, сгенерированный `Агентом`, соответствует основным паттернам и стилю целевого `Проекта`.
- `Агент` корректно обрабатывает ошибки и предоставляет информативные логи.
- Документация по установке, настройке и использованию `Агента` предоставлена.