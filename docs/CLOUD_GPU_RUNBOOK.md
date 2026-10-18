# Runpod Secure Cloud: нативный запуск CashWatch

После сообщения о RTX 4050 у второго Pro сначала измеряем имеющийся ноутбук по [RTX_4050.md](RTX_4050.md). Этот платный путь остаётся резервом; аренда и бюджет ещё не подтверждены.

Дата проверки публичной документации: **17 октября 2026 года**. Этот документ описывает запуск собственного Python-сервиса с локальными весами внутри GPU Pod после его выделения команде. Аренда, создание аккаунта, пополнение баланса и развёртывание в рамках этой работы не выполнялись. Предложение бюджета **35 USD пока не одобрено**.

Учебный архив уже получен из приложенных пользователем материалов и обработан локально: 42 MOV, табличные метки 22 positive / 20 normal, 40 уникальных SHA. Функциональное обучение DINO/head на четырёх разных видео завершилось на Linux CPU; hosted AI для банковских материалов не использовался. Это отдельный факт из [FUNCTIONAL_TRAINING.md](FUNCTIONAL_TRAINING.md), а не выполненное развёртывание Runpod, CUDA benchmark или независимая оценка качества. Банковские файлы на арендуемую GPU-машину не переносились.

## 1. Выбор и границы решения

Для GPU-варианта выбираем **Secure Cloud, один GPU, On-Demand, официальный PyTorch template**. Точный доступный GPU, образ, tag, версия Python, регион, цена выбранной машины и объём памяти должны быть зафиксированы по фактически выбранному Pod. Доступность RTX 4090 или L4 не проверена. Слово Secure само по себе не подтверждает выполнение всех требований к данным.

Это собственный сервис команды: веса загружаются заранее, анализ выполняет локальный процесс. Готовые Runpod Public Endpoints, облачные LLM/API и пересылка им банковских видео, кадров или описаний в этой схеме не используются. Такое прочтение требования локальности обосновано текстом кейса; отдельного прямого разрешения на арендованное железо в кейсе не найдено. Разбор точных формулировок: [HOSTING_DECISION.md](HOSTING_DECISION.md).

**Docker Compose внутри Runpod Pod не поддерживается.** Сам Pod уже запускается из контейнерного образа; следующие команды устанавливают Python-окружение и запускают Uvicorn непосредственно в нём. Инструкция Compose для обычного собственного Linux-сервера находится отдельно в [DEPLOYMENT.md](DEPLOYMENT.md). [Официальные ограничения Pods](https://docs.runpod.io/pods/overview).

Текущий комплект кода ещё не является готовым конкурсным развёртыванием: три нормативных правила выключены, запуск этой последовательности на Runpod не выполнен, внешний URL отсутствует. Загрузка весов, зелёный healthcheck и наличие GPU не доказывают качество анализа или соблюдение пяти минут.

## 2. Стоимость и срок работы

На публичной [странице цен Runpod](https://www.runpod.io/pricing), помеченной Updated October 8, 2026, в секции **Pods** видны следующие ориентиры. Текстовое представление одновременно показывает переключатели Community/Secure; выбранный режим не установлен, поэтому это **не подтверждённые предложения Secure Cloud**.

| Ориентир | Публичная цена Pods | Плановые 30 часов, только GPU |
|---|---:|---:|
| RTX 4090, 24 GB VRAM | 0,89 USD/час | 26,70 USD |
| L4, 24 GB VRAM | 0,59 USD/час | 17,70 USD |

Это не цены Serverless и не оплата готовой модели за запрос. Фактическая стоимость определяется выбранным Pod в консоли. 30 часов — расчётный интервал с запасом на проверку жюри, не обещание времени обработки. Сдача по условию пользователя: **18 октября 2026, 12:00 Asia/Almaty (UTC+5), то есть 07:00 UTC**. Официальное [расписание события](https://datathon.com.kz/schedule) указывает питчинг 13:00–17:00 и закрытие 18:00–19:00 в тот же день; расписание может изменяться.

| Хранилище | При работающем Pod, USD/GB/месяц | После Stop | После Terminate |
|---|---:|---|---|
| Container disk | 0,10 | Данные удаляются, оплата этого диска прекращается | Удаляется |
| Обычный volume disk | 0,10 | Данные остаются; 0,20 USD/GB/месяц | Данные удаляются |
| Стандартный network volume менее 1 TB | 0,07 | Сохраняется и оплачивается отдельно | Сохраняется и оплачивается отдельно |

Для network volume от 1 TB документация указывает 0,05 USD/GB/месяц. High-performance storage — отдельный продукт; в этот расчёт он не включён. Container/volume disk тарифицируются посекундно, network volume — почасово. [Тарифы Pods и хранения](https://docs.runpod.io/pods/pricing).

Для первичного плана можно рассмотреть 20 GB container disk и 60 GB обычного volume disk. **Это предложенный размер, фактический расход ещё не измерен.** При условном 30-дневном месяце их совместная плата за 30 часов работы составила бы около 0,33 USD; ориентир RTX 4090 вместе с таким хранением — около 27,03 USD до дополнительных часов и применимых начислений. 35 USD — запас для планирования, а не установленный лимит списаний. До выделения ресурсов нужно увидеть итоговую цену выбранной конфигурации. Остановка Python-процесса не останавливает оплату Pod; нулевой баланс не используем как механизм безопасного завершения.

## 3. Параметры уже выбранного Pod

Следующая процедура начинается **после отдельного решения о расходах и появления реального Pod**. Она не создаёт платные ресурсы.

| Параметр | Что требуется для этой инструкции |
|---|---|
| Среда | Linux x86-64, Ubuntu/Debian, Bash; официальный PyTorch template |
| Python | Проект требует ≥3.11; предпочтителен 3.12, на котором проверялся основной комплект зависимостей |
| ML | Закреплённые проектом Torch 2.8.0, torchvision 0.23.0, Transformers 4.57.1; полный список в `pyproject.toml` |
| GPU | Один реально выделенный CUDA GPU; доступная VRAM и работоспособность проверяются командами ниже |
| CUDA | Здесь выбран официальный wheel PyTorch для CUDA 12.8; хост и драйвер должны его поддерживать |
| HTTP | Expose HTTP Port: **8000** |
| Передача кода | Full SSH через public IP и фактический mapped TCP port; ключ SSH уже настроен |
| Данные | Persistent mount **`/workspace`**, подтверждённый настройками конкретного Pod |

Официальные templates содержат настройки образа, дисков, портов и запуска. Их состав необходимо проверять; пример tag из документации не означает его текущую доступность. [Templates](https://docs.runpod.io/pods/templates/overview), [официальный пример PyTorch template](https://docs.runpod.io/pods/templates/create-custom-template).

Обычный volume disk по умолчанию монтируется в `/workspace` и сохраняется при Stop, но привязан к Pod. Network volume заменяет его в той же точке и живёт независимо; подключать его нужно при создании. Для обычного volume доступна настройка Encrypt volume; по документации она не шифрует container disk и network volume. Не считайте наличие шифрования доказательством отсутствия доступа инфраструктурного провайдера. [Свойства хранения](https://docs.runpod.io/pods/storage/types).

Всё, что нужно сохранить при перезапуске, размещаем в `/workspace`: код, venv, веса, конфигурацию и локальные отчёты. Установленные через apt системные пакеты находятся вне этого тома: после Stop/Edit необходимо повторить проверку среды. Venv также зависит от базового Python образа и не переносится между произвольными образами без пересоздания. [Python venv](https://docs.python.org/3.12/library/venv.html).

## 4. Передача только архива исходников

Этот блок выполняется **на Mac**, в Bash. Значения берутся из реальной вкладки Connect → SSH over exposed TCP. Публичный GitHub-репозиторий команды пока не найден, поэтому здесь нет выдуманной команды `git clone`.

В архиве кода должна быть верхняя папка `forte-datathon`. Используйте подготовленный архив проекта без `.env`, банковских файлов и результатов анализа. `open.zip` с конкурсными материалами не является архивом исходников.

```bash
set -euo pipefail
read -r -p 'Абсолютный путь к архиву исходников: ' CASHWATCH_SOURCE_ZIP
read -r -p 'Public IP из SSH over exposed TCP: ' CASHWATCH_SSH_HOST
read -r -p 'Внешний TCP port SSH из Connect: ' CASHWATCH_SSH_PORT
read -r -p 'Пользователь SSH из Connect: ' CASHWATCH_SSH_USER
read -r -p 'Абсолютный путь к приватному SSH-ключу: ' CASHWATCH_SSH_KEY
test -f "$CASHWATCH_SOURCE_ZIP"
test -f "$CASHWATCH_SSH_KEY"
scp -P "$CASHWATCH_SSH_PORT" -i "$CASHWATCH_SSH_KEY" \
  "$CASHWATCH_SOURCE_ZIP" \
  "$CASHWATCH_SSH_USER@$CASHWATCH_SSH_HOST:/workspace/cashwatch-source.zip"
ssh -p "$CASHWATCH_SSH_PORT" -i "$CASHWATCH_SSH_KEY" \
  "$CASHWATCH_SSH_USER@$CASHWATCH_SSH_HOST"
```

Обычное подключение через `ssh.runpod.io` не поддерживает SCP/SFTP; для этого блока требуется именно Full SSH. Не отключайте проверку SSH host key. [Официальная инструкция SSH](https://docs.runpod.io/pods/configuration/use-ssh).

## 5. Чистое окружение: команды внутри Pod

Все последующие блоки до проверки внешнего URL выполняются **в одной Bash-сессии Pod**. При ошибке остановитесь на конкретном шаге; не переходите к обработке банковских видео с непрошедшей проверкой.

### 5.1. ОС, интерпретатор, том и системные пакеты

```bash
set -euo pipefail
umask 077
test "$(uname -s)" = Linux
test "$(uname -m)" = x86_64
test -d /workspace
df -h /workspace
CASHWATCH_BASE_PYTHON="$(command -v python3)"
"$CASHWATCH_BASE_PYTHON" - <<'PY'
import json, sys
if sys.version_info < (3, 11):
    raise SystemExit('This project requires Python >=3.11; use a compatible official template.')
print(json.dumps({'python': sys.version.split()[0], 'executable': sys.executable}))
PY

if [ "$(id -u)" -eq 0 ]; then
  CASHWATCH_SUDO=()
elif command -v sudo >/dev/null; then
  CASHWATCH_SUDO=(sudo)
else
  printf '%s\n' 'apt installation needs root or sudo in this selected template.' >&2
  exit 1
fi
"${CASHWATCH_SUDO[@]}" apt-get update
"${CASHWATCH_SUDO[@]}" apt-get install --yes --no-install-recommends \
  ffmpeg libseccomp2 ca-certificates python3-venv
command -v ffmpeg
command -v ffprobe
command -v timeout
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

test ! -e /workspace/forte-datathon
"$CASHWATCH_BASE_PYTHON" -m zipfile -e /workspace/cashwatch-source.zip /workspace
export CASHWATCH_ROOT=/workspace/forte-datathon
cd "$CASHWATCH_ROOT"
test -f pyproject.toml
test -f src/cashwatch/network_guard.py
mkdir -p local_reports runtime models
chmod 700 local_reports runtime models
```

Неудача `nvidia-smi` или отсутствие назначенного GPU — незакрытая проблема выделенной среды. Она не исправляется переключением проекта на CPU с сохранением заявления о проверенной скорости GPU.

### 5.2. Изолированный venv и закреплённые ML-пакеты

Здесь создаётся чистый venv; пакеты template автоматически не наследуются. CUDA wheel устанавливается по [официальной инструкции PyTorch 2.8.0](https://pytorch.org/get-started/previous-versions/). Точный базовый путь Python обнаружен на предыдущем шаге. Не меняйте версии Torch и torchvision независимо друг от друга.

```bash
test ! -e /workspace/cashwatch-venv
"$CASHWATCH_BASE_PYTHON" -m venv /workspace/cashwatch-venv
export CASHWATCH_PYTHON=/workspace/cashwatch-venv/bin/python
export PIP_NO_CACHE_DIR=1
"$CASHWATCH_PYTHON" -m pip install -r requirements-core.lock.txt
"$CASHWATCH_PYTHON" -m pip install \
  torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
"$CASHWATCH_PYTHON" -m pip install -e '.[ml,test,training]' \
  -c requirements-core.lock.txt
"$CASHWATCH_PYTHON" -m pip check
"$CASHWATCH_PYTHON" -m pip freeze > local_reports/pod-packages.local.txt
"$CASHWATCH_PYTHON" - <<'PY'
import json, torch, torchvision, transformers
if not torch.cuda.is_available():
    raise SystemExit('CUDA is unavailable in this exact Python environment.')
x = torch.ones((2, 2), device='cuda')
_ = x @ x
torch.cuda.synchronize()
print(json.dumps({
    'torch': torch.__version__, 'torchvision': torchvision.__version__,
    'transformers': transformers.__version__, 'cuda_runtime': torch.version.cuda,
    'gpu': torch.cuda.get_device_name(0),
    'vram_bytes': torch.cuda.get_device_properties(0).total_memory,
    'cuda_tensor_check': True,
}))
PY
```

Если CUDA wheel несовместим с выделенным хостом, эту ошибку нужно решить до модели. Инструкция не устанавливает драйвер хоста и не доказывает доступность конкретного GPU. Не заменяйте закреплённые версии случайными пакетами ради успешного импорта; изменённая комбинация требует повторной проверки.

### 5.3. Linux network guard — обязательная проверка до модели

```bash
"$CASHWATCH_PYTHON" -m cashwatch.network_guard --check \
  > local_reports/network-guard.local.json
"$CASHWATCH_PYTHON" - <<'PY'
import json
from pathlib import Path
r = json.loads(Path('local_reports/network-guard.local.json').read_text())
required = ('ok', 'os_enforced', 'ipv4_socket_denied',
            'ipv6_socket_denied', 'unix_socketpair_works')
if not all(r.get(key) is True for key in required):
    raise SystemExit('The Linux network guard did not pass; inference must stay blocked.')
print(json.dumps({key: r[key] for key in required}))
PY
```

CLI проверяет фильтр в отдельнем одноразовом subprocess; сеть текущего shell и HTTP API сохраняется. Worker устанавливает libseccomp-фильтр до загрузки ML и отказывается продолжать при ошибке. Запрещено создание обычных IPv4/IPv6 sockets; AF_UNIX остаётся доступным. Не передавайте worker сетевой proxy через UNIX socket. Это узкая изоляция доверенного процесса и его потомков, а не firewall всего хоста или sandbox произвольного вредоносного кода. Фактический контракт: [`network_guard.py`](../src/cashwatch/network_guard.py), [`worker.py`](../src/cashwatch/worker.py).

На Mac этот Linux-механизм не действует: offline-переменные и локальные веса не являются доказательством системного запрета сети. Не отключайте Linux guard, чтобы обойти несовместимость Pod.

### 5.4. Публичные веса: подготовка с сетью, затем offline smoke

Этот downloader обращается к публичному репозиторию модели, проверяет лицензию, фиксирует commit и SHA-256; банковские материалы он не принимает. Если веса уже подготовлены, переносите весь каталог вместе с `cashwatch_model_manifest.json` и проверяйте хеши.

```bash
env HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
  "$CASHWATCH_PYTHON" scripts/model_download.py --model qwen3-vl-2b \
  --output "$CASHWATCH_ROOT/models/Qwen3-VL-2B-Instruct"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
timeout --signal=TERM --kill-after=10s 240s "$CASHWATCH_PYTHON" - \
  > local_reports/cuda-smoke.local.json <<'PY'
import runpy, sys
from cashwatch.network_guard import enforce_network_guard
r = enforce_network_guard()
if r.get('os_enforced') is not True:
    raise SystemExit('OS network isolation is required for this Linux smoke test.')
sys.argv = [
    'scripts/check_model.py', '--model-dir', 'models/Qwen3-VL-2B-Instruct',
    '--device', 'cuda', '--verify-hashes', '--smoke', '--seconds', '120',
]
runpy.run_path('scripts/check_model.py', run_name='__main__')
PY
"$CASHWATCH_PYTHON" - <<'PY'
import json
from pathlib import Path
r = json.loads(Path('local_reports/cuda-smoke.local.json').read_text())
if r.get('weights_loaded') is not True or r.get('real_generation_tested') is not True:
    raise SystemExit('The model smoke test did not confirm loading and generation.')
print(json.dumps({key: r.get(key) for key in (
    'weights_loaded', 'real_generation_tested', 'device',
    'load_seconds', 'real_competition_video_tested')}))
PY
```

Smoke использует созданное программой пустое изображение. Он проверяет загрузку и генерацию под запретом сети, но не качество на кассовых событиях и не скорость полного видео. Код возврата 124 от `timeout` означает, что проверка не завершилась за заданный предел; это не успешный тест. Значение 240 секунд относится только к этому диагностическому запуску.

### 5.5. Конфигурация сервиса без вывода ключа

```bash
test -f .env || "$CASHWATCH_PYTHON" scripts/init_env.py
"$CASHWATCH_PYTHON" - <<'PY'
from pathlib import Path
from dotenv import set_key
p = Path('.env')
values = {
    'CASHWATCH_MODEL_DIR': '/workspace/forte-datathon/models/Qwen3-VL-2B-Instruct',
    'CASHWATCH_DATA_DIR': '/workspace/forte-datathon/runtime',
    'CASHWATCH_RULES_PATH': '/workspace/forte-datathon/config/rules.json',
    'CASHWATCH_DEVICE': 'cuda', 'CASHWATCH_MAX_INFLIGHT': '1',
    'CASHWATCH_TIMEOUT_SEC': '275',
    'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
    'HF_HUB_DISABLE_TELEMETRY': '1', 'DO_NOT_TRACK': '1',
}
for key, value in values.items():
    set_key(str(p), key, value, quote_mode='always')
p.chmod(0o600)
print('Saved configuration; the API key was not printed.')
PY
"$CASHWATCH_PYTHON" -m pytest -q
```

Тесты используют синтетические материалы. Отчёт pytest в этом Pod нужно сохранить фактически; результат из другой машины не подтверждает эту среду. Число тестов намеренно не зафиксировано в инструкции, поскольку комплект меняется.

Все три правила в `config/rules.json` по умолчанию `enabled=false`. Для анализа требуется локально проверить применимость и визуальные условия каждого включаемого правила по доступным материалам и документам. Этот runbook правила не включает. При отсутствии активных правил сервис должен явно отклонять анализ; нельзя получать фиктивный отрицательный вердикт, просто запустив сервер. См. [REGULATIONS.md](REGULATIONS.md) и [UNSAFE_ASSUMPTIONS.md](UNSAFE_ASSUMPTIONS.md).

## 6. Запуск HTTP-процесса

Сначала можно проверить интерфейс и диагностику даже с выключенными правилами. Это ещё не конкурсная приёмка. Ниже один Uvicorn worker: менеджер задач и лимит конкуренции находятся в процессе API.

```bash
cat > /workspace/cashwatch-start.sh <<'BASH'
#!/usr/bin/env bash
set -euo pipefail
umask 077
cd /workspace/forte-datathon
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1
export CASHWATCH_DEVICE=cuda CASHWATCH_MAX_INFLIGHT=1
/workspace/cashwatch-venv/bin/python -m cashwatch.network_guard --check >/dev/null
exec /workspace/cashwatch-venv/bin/python -m uvicorn \
  cashwatch.main:app_factory --factory --host 0.0.0.0 --port 8000 --workers 1
BASH
chmod 700 /workspace/cashwatch-start.sh
nohup /workspace/cashwatch-start.sh > local_reports/server.local.log 2>&1 &
printf '%s\n' "$!" > local_reports/server.local.pid
```

Откройте `http://127.0.0.1:8000/healthz` из Pod после появления процесса. Проверка без печати ключа:

```bash
"$CASHWATCH_PYTHON" - <<'PY'
import json, os, urllib.error, urllib.request
from dotenv import load_dotenv
load_dotenv('.env', override=False)
base = 'http://127.0.0.1:8000'
for path, authenticated in (('/healthz', False), ('/api/v1/capabilities', True)):
    headers = {'X-API-Key': os.environ['CASHWATCH_API_KEY']} if authenticated else {}
    req = urllib.request.Request(base + path, headers=headers)
    with urllib.request.urlopen(req, timeout=10) as response:
        payload = json.load(response)
        print(json.dumps({'path': path, 'http_status': response.status,
                          'payload': payload}, ensure_ascii=False))
PY
```

Capabilities сообщает о конфигурации; `ready_for_submission` не является автоматически включаемым флагом готовности. Наличие весов не закрывает проверку закона, качества, времени и доступности.

`nohup` переживает отключение SSH, но не Stop/Terminate Pod. После перезапуска проверьте базовую среду, системные пакеты, GPU, guard и запуск. Не запускайте второй экземпляр поверх первого. Если исходная сессия закрылась, переменные восстанавливаются явно:

```bash
export CASHWATCH_ROOT=/workspace/forte-datathon
export CASHWATCH_PYTHON=/workspace/cashwatch-venv/bin/python
cd "$CASHWATCH_ROOT"
```

## 7. HTTPS URL и проверка с другой сети

В Connect выберите HTTP service на порту 8000. Официальная форма URL:

```text
https://<фактический-POD_ID>-8000.proxy.runpod.net
```

Скопируйте выданный URL; идентификатор и готовый адрес здесь не выдуманы. API слушает `0.0.0.0`, внутри Pod используется HTTP, внешний proxy предоставляет HTTPS. Это публичный маршрут: авторизация `X-API-Key` остаётся обязательной. [Порты и proxy Runpod](https://docs.runpod.io/pods/configuration/expose-ports).

У этого proxy **100-секундный timeout соединения**. Поэтому длинный анализ выполняется после быстрого `202`, а результат получается короткими polling-запросами. Ограничение конкурса 300 секунд относится ко всему пути, включая загрузку. `jury_check.py` дополнительно ограничивает POST 60 секундами. Нужно измерить передачу реального файла через внешний URL: максимальный размер тела и поведение буферизации выбранного proxy пока не проверены. Ответ 524, поздний 202 или успешный localhost не подтверждают доступность для жюри.

Следующий блок выполняется **на Mac**, где уже есть окружение проекта, разрешённые локальные видео, `.env` с ключом именно развёрнутого сервиса и корректный manifest. Подставляется только HTTPS origin собственного сервиса команды. Команда отправляет видео на этот сервис; не направляйте её на готовый AI endpoint.

```bash
read -r -p 'HTTPS origin собственного сервиса из Connect: ' CASHWATCH_PUBLIC_ORIGIN
python scripts/jury_check.py \
  --base-url "$CASHWATCH_PUBLIC_ORIGIN" \
  --manifest data/manifest.local.json --split validation \
  --out local_reports/runpod-validation-predictions.local.json
python scripts/evaluate.py \
  --manifest data/manifest.local.json \
  --predictions local_reports/runpod-validation-predictions.local.json \
  --split validation --out local_reports/runpod-validation-metrics.local.json
```

До этого прогона нужно завершить разметку и независимое разделение источников; неизвестный `group_id=unassigned` не доказывает отсутствие утечки. Holdout используется один раз после фиксации модели, параметров и правил. Отчёты содержат чувствительные результаты и кадры доказательств: сохраняйте их локально, не публикуйте в GitHub. Подробности метрик и ограничений: [DATA_AND_EVALUATION.md](DATA_AND_EVALUATION.md).

## 8. Что должно быть реально подтверждено до передачи URL

| Проверка | Нужное свидетельство | Статус на момент подготовки инструкции |
|---|---|---|
| Выделение и расходы | Реальный Secure Cloud Pod, регион, image/tag, цена и согласованный бюджет | Не выполнено |
| Окружение | Python/torch/torchvision/CUDA, успешные `pip check` и GPU tensor check | На Runpod не проверено |
| Network guard | Exit 0, deny IPv4/IPv6, рабочий AF_UNIX; успешный CUDA smoke под guard | На Runpod не проверено |
| Веса Qwen на Runpod | Локальный commit, SHA-256, загрузка и генерация | На Runpod не выполнено; локальный CPU-прогон DINO не заменяет эту проверку |
| Применимые нормы | Проверенные условия включённых правил, исключения и источники | Все три правила выключены |
| Конкурсное качество | Все validation-предсказания, failed отдельно, независимый holdout после фиксации | Не измерено этой инструкцией |
| Полный внешний путь | MOV → HTTPS POST 202 → polling → терминальный результат с доказательствами | Не проверено |
| Пять минут | Измеренный wall-clock на длинном видео и холодном старте | Таймаут 275 секунд не заменяет измерение |
| Жизненный цикл | Последовательные запросы, перегрузка, неверный ключ, рестарт и TTL | Нужна проверка целевого Pod |
| Передача команде/жюри | Реально доступный URL и приватный ключ через согласованный канал | Не выполнялась |

## 9. Окончание платного интервала

Сервис должен оставаться доступным в согласованный период проверки. Перед завершением сохраните нужные исходники, lock и обезличенные измерения в разрешённом командном месте; требования к банковским материалам рассматриваются отдельно. TTL API не удаляет загруженный учебный архив, каталог экспериментов, скачанные отчёты и модельные веса.

**Stop Pod** прекращает использование GPU, но обычный volume disk продолжает оплачиваться. Возобновление может получить zero GPUs, если свободная ёмкость изменилась. **Terminate Pod** необратимо удаляет обычный volume; network volume, если выбран, остаётся отдельным оплачиваемым ресурсом. После завершения проверьте фактическое состояние ресурсов и начислений в консоли. Команды аренды, автоплатежа и автоматического удаления здесь намеренно не выполнялись. [Жизненный цикл Pod](https://docs.runpod.io/pods/manage-pods).

Runbook подтверждает подготовленную процедуру. Протокол фактического выполнения с датой, конфигурацией и результатами должен быть добавлен после запуска на выбранной машине.
