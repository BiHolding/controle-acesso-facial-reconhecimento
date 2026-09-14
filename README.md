# Controle de acesso facial — integrado ao Portal VIP

Cliente local do computador da entrada da Sala VIP. Captura uma face, gera um
embedding com InsightFace e solicita a decisão à API oficial do Portal VIP. A
mesma API identifica o participante, aplica as regras e registra entrada/saída.

## Fluxo

```text
Câmera -> InsightFace buffalo_l -> embedding L2 (512 floats)
  -> controle de qualidade e média de 3 amostras estáveis
  -> identificação local preparada para contingência
  -> API VIP /access-control/recognize
  -> matching no participant_face_embeddings
  -> regras de acesso + ocupação atômica
  -> allowed/denied -> badge local
```

API indisponível e sem snapshot local válido = FAIL CLOSED (badge amarelo
"NÃO FOI POSSÍVEL VALIDAR").

Quando existe um snapshot local válido, a queda da API ativa automaticamente o
modo de contingência: a decisão usa o cache criptografado do Windows, atualiza a
ocupação local compartilhada entre as duas estações e entra em uma fila FIFO. A
estação principal reenvia a fila na ordem original assim que a API responder.
Embeddings nunca são gravados em texto aberto e a fila contém apenas IDs técnicos.

## Requisitos

- Python 3.12+
- Camera conectada
- API VIP acessível por HTTPS
- MySQL/FTP acessíveis somente na estação de enrollment automático

## Instalacao

```bash
python -m venv .venv
python -m pip install -e .
```

No Windows PowerShell, ative com `.venv\Scripts\Activate.ps1`.

## Configuracao

Crie `.env` a partir de `.env.example`:

```env
DB_HOST=127.0.0.1
DB_PORT=3306
DB_NAME=ispevolution_p
DB_USER=replace-with-db-user
DB_PASSWORD=replace-with-db-password

API_URL=https://ispevolution.com.br/api/v1
ACCESS_POINT=ENTRADA_PRINCIPAL
DEVICE_KEY=replace-with-device-key
ENROLLMENT_DEVICE_KEY=replace-with-enrollment-key
FACE_CONFIRMATION_SAMPLES=3
VIP_MAX_CAPACITY=400
```

## Estrutura

```
src/reconhecimento/
├── camera/
│   └── capture.py          # Camera via OpenCV
├── database/
│   └── repository.py       # MySQL: convidados pendentes de enrollment
├── recognition/
│   ├── detector.py         # Deteccao de rostos (InsightFace buffalo_l, detection only)
│   ├── embedder.py         # Geracao de embedding L2 (ArcFace ONNX direto)
│   └── quality.py          # iluminação, nitidez e enquadramento
├── api/
│   └── client.py           # Reconhecimento e enrollment na API VIP
└── recognize.py            # Loop principal
```

## Execucao

```bash
reconhecer
```

Ou `python -m reconhecimento.recognize`. Pressione `q` para encerrar.

### Duas webcams e dois monitores no mesmo notebook

Configure o pareamento no `.env`:

```env
STATION_1_CAMERA_INDEX=0
STATION_1_DISPLAY_INDEX=0
STATION_1_DEVICE_ID=0
STATION_1_DIRECTION=ENTRY
STATION_1_ACCESS_POINT=ENTRADA_PRINCIPAL
STATION_1_DEVICE_KEY=replace-with-entry-device-key
STATION_2_CAMERA_INDEX=1
STATION_2_DISPLAY_INDEX=1
STATION_2_DEVICE_ID=0
STATION_2_DIRECTION=EXIT
STATION_2_ACCESS_POINT=SAIDA_PRINCIPAL
STATION_2_DEVICE_KEY=replace-with-exit-device-key
```

Depois inicie as duas estações com:

```bash
reconhecer-duplo
```

Cada estação roda em um processo isolado. Se uma câmera apresentar falha, a
outra permanece disponível. O inicializador bloqueia câmera ou monitor duplicado
e avisa quando o segundo monitor não está habilitado no Windows. Para registrar
as decisões na API, configure uma chave técnica diferente para cada estação.
O enrollment automático roda somente na estação
1, evitando que as duas instâncias processem a mesma fotografia simultaneamente.
Os blocos `STATION_1_*` e `STATION_2_*` podem ser invertidos livremente: webcam,
monitor, função e ponto de acesso sempre permanecem associados no mesmo bloco.

Estados da UI:

- Verde: acesso autorizado e nome.
- Vermelho: acesso nao autorizado.
- Vermelho: pessoa nao identificada.
- Amarelo: nao foi possivel validar.

## Modelo e confirmacao

- Pack InsightFace: `buffalo_l`.
- Provider: `CPUExecutionProvider`.
- Modulos: deteccao (FaceAnalysis) + reconhecimento (ArcFace ONNX direto).
- Entrada do detector: 320x320.
- Embedding esperado: 512 dimensoes, normalizado em L2.
- Comparação: produto escalar/cosseno no backend contra o mesmo banco do Portal VIP.
- Confirmação local: média normalizada de 3 amostras boas antes de uma única chamada à API.
- Cooldown local: 10 segundos por Guest e 10 segundos para desconhecido.

O reconhecimento e 100% local. Nao existe chamada HTTP para reconhecimento.
A revalidacao MySQL e feita antes de `allowed=true`. DB indisponivel = negado.

## Elegibilidade

Guest elegivel: `status=completed` AND `client.status=active`.
Embedding elegivel: `active=1`, `model=buffalo_l`, `dimension=512`, `normalization=l2`.
Sync periodico: a cada 30 segundos (configuravel).

## BLOB do embedding

O MySQL armazena o embedding como BLOB (PHP `pack('g*')`) = 2048 bytes =
512 float32 little-endian nativo. O Python le com `numpy.frombuffer()`.

## Testes

```bash
python -m unittest discover -s tests -v
```

Detalhes da integracao e das mudancas requeridas no backend estao em
`docs/integracao-portal-vip.md`.
