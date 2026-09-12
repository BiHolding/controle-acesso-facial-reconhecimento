# Controle de acesso facial — local-first

Cliente local do computador da entrada da Sala VIP. Captura uma face, gera um
embedding com InsightFace e faz matching local contra o banco MySQL. Não existe
servidor facial separado. O reconhecimento é 100% local.

## Fluxo

```text
MySQL (guests + client + guest_face_embeddings)
  -> Python carrega embeddings elegiveis
  -> FaceIndex NumPy local (busca cosseno exata)
Câmera -> InsightFace buffalo_l -> embedding L2 (512 floats)
  -> matching local -> threshold
  -> 5 confirmacoes consecutivas
  -> revalidacao MySQL (guest completed + client active)
  -> allowed/denied -> badge local -> access_event
```

DB indisponivel = FAIL CLOSED (badge amarelo "NAO FOI POSSIVEL VALIDAR").

## Requisitos

- Python 3.12+
- Camera conectada
- MySQL acessivel (ispevolution_p)
- Credenciais de banco com privilegios minimos

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

ACCESS_POINT=vip_room
DEVICE_ID=0

FACE_MATCH_THRESHOLD=0.60
FACE_MODEL_VERSION=v1
FACE_SYNC_INTERVAL_SECONDS=30
```

`DEVICE_ID=0` desabilita gravacao de access_events.

## Estrutura

```
src/reconhecimento/
├── camera/
│   └── capture.py          # Camera via OpenCV
├── database/
│   └── repository.py       # MySQL: embeddings, revalidation, access_events
├── recognition/
│   ├── detector.py         # Deteccao de rostos (InsightFace buffalo_l, detection only)
│   ├── embedder.py         # Geracao de embedding L2 (ArcFace ONNX direto)
│   ├── matcher.py          # FaceIndex NumPy (busca cosseno exata)
│   ├── confirmation.py     # 5 observacoes consecutivas
│   └── guard.py            # Cooldown por pessoa
├── sync/
│   └── face_sync.py        # Sincronizacao periodica MySQL -> index
├── api/
│   └── client.py           # Legado (nao usado no fluxo local)
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
STATION_1_ACCESS_POINT=VIP_ENTRANCE_01
STATION_2_CAMERA_INDEX=1
STATION_2_DISPLAY_INDEX=1
STATION_2_DEVICE_ID=0
STATION_2_DIRECTION=EXIT
STATION_2_ACCESS_POINT=VIP_EXIT_01
```

Depois inicie as duas estações com:

```bash
reconhecer-duplo
```

Cada estação roda em um processo isolado. Se uma câmera apresentar falha, a
outra permanece disponível. O inicializador bloqueia câmera ou monitor duplicado
e avisa quando o segundo monitor não está habilitado no Windows. Para registrar
os eventos no banco, substitua os zeros por dois IDs válidos e distintos da
tabela `access_control_devices`. O enrollment automático roda somente na estação
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
- Comparacao local: produto escalar/cosseno entre vetores normalizados.
- Confirmacao: 5 observacoes consecutivas da mesma identidade.
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
