# controle-acesso-facial-reconhecimento

Cliente local de reconhecimento facial.
Captura frames da câmera, detecta rostos, gera embeddings e envia para a API hospedada.
A API cuida do banco, da busca por similaridade e do registro dos eventos.

---

## Como funciona

```
Câmera → detecta rosto → gera embedding (512 floats)
    → POST /api/recognize na API
        → API busca no banco via pgvector
        → API verifica permissão
        → API salva o log
    → Badge verde (permitido) ou vermelho (negado) na tela
```

---

## Requisitos

- Python 3.12+
- Câmera conectada (USB ou embutida)
- Acesso à API hospedada (URL + chave)

---

## Instalação

```bash
git clone <url-do-repositorio>
cd controle-acesso-facial-reconhecimento

python3 -m venv env
source env/bin/activate

pip install -e .
```

---

## Configuração

```bash
cp .env.example .env
```

Edite o `.env`:

```env
# URL base da API hospedada (sem barra no final)
API_URL=https://meu-servidor.com

# Deve ser igual ao API_KEY configurado no servidor
API_KEY=troque-por-uma-chave-segura

# Ponto de acesso desta máquina (main_door ou vip_room)
ACCESS_POINT=main_door
```

---

## Rodando

```bash
source env/bin/activate
reconhecer
```

Ou diretamente:

```bash
python -m reconhecimento.recognize
```

Pressione `Q` para encerrar.

---

## Estrutura

```
src/reconhecimento/
├── camera/
│   └── capture.py          # Câmera via OpenCV
├── recognition/
│   ├── detector.py         # Detecção de rostos (InsightFace buffalo_l)
│   ├── embedder.py         # Geração de embedding L2-normalizado
│   ├── confirmation.py     # Janela deslizante — evita falsos positivos
│   └── guard.py            # Cooldown por pessoa — evita eventos duplicados
├── api/
│   └── client.py           # Chama POST /api/recognize via httpx
└── recognize.py            # Loop principal
```

---

## Ajustes de comportamento

Edite as constantes no topo de `src/reconhecimento/recognize.py`:

| Constante | Padrão | Descrição |
|---|---|---|
| `PROCESS_EVERY_N` | `3` | Analisa 1 a cada N frames (alivia CPU) |
| `SHOW_SECS` | `3.0` | Segundos que o badge fica visível |
| `UNKNOWN_REQUIRED` | `10` | Frames sem reconhecimento para mostrar "Desconhecido" |

A janela de confirmação (`required_frames=5`) e o cooldown por pessoa (`cooldown_seconds=10`) também podem ser ajustados no `recognize.py`.
