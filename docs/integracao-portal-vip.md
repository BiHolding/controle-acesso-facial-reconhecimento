# Integração com o Portal VIP ISP Evolution

## Decisão arquitetural

O Portal VIP permanece a fonte de verdade. O computador da porta envia somente
um embedding e recebe somente `recognized`, `allowed`, `guestId`, `name`,
`similarity` e `reason`. Não recebe foto, CPF, telefone, email ou endereço.

A primeira versão é online e fail-closed. Não existe cache autorizador local.
Uma indisponibilidade exige validação manual e nunca libera acesso.

## Componentes

```text
Portal CI4/MySQL
  Guest completed + Client active + foto privada
       |
       | endpoint M2M de foto (somente enrollment service)
       v
Serviço facial Python
  InsightFace buffalo_l -> embedding L2 -> MySQL guest_face_embeddings
  índice NumPy em memória, reconstruído/sincronizado por versão da foto
       ^
       | POST access-control/recognize (embedding, accessPoint)
       |
PC da porta
  câmera -> embedding -> resultado -> confirmação/cooldown -> UI
```

Para volume pequeno/médio, busca cosseno exata em uma matriz NumPy é a opção
mais simples. MySQL persiste os templates e metadados; não é necessário
PostgreSQL, pgvector ou outro banco na primeira versão.

## Endpoints M2M

### Reconhecimento

`POST /api/v1/access-control/recognize`

Requer `X-Device-Key`, limita payload e taxa, valida 512 números finitos e norma
L2. O `accessPoint` deve coincidir com o ponto vinculado ao dispositivo. A API:

1. encontra o vizinho mais próximo acima de `FACE_MATCH_THRESHOLD`;
2. relê o Guest e o Client no MySQL;
3. exige Guest existente e `status=completed`;
4. exige Client existente e `status=active`;
5. aplica regras adicionais de elegibilidade quando forem definidas;
6. retorna somente os campos mínimos do contrato.

Razões de decisões válidas: `AUTHORIZED`, `UNKNOWN_FACE` e `NOT_AUTHORIZED`.
Índice ausente ou inválido retorna HTTP 503 com código
`FACE_INDEX_UNAVAILABLE`; o cliente mostra erro e nega.

### Foto privada para enrollment

`GET /api/v1/access-control/guests/{id}/photo`

Este endpoint é exclusivo de uma credencial com escopo `face:enroll`, não da
credencial da porta. Deve carregar a referência pelo ID no banco, exigir Guest
completed e Client active, resolver o caminho sob `WRITEPATH/guests`, conferir
com `realpath` que não houve traversal e responder a imagem com MIME persistido,
`Cache-Control: private, no-store` e `X-Content-Type-Options: nosniff`.

Não aceita referência/path fornecido pelo chamador e não cria URL permanente.
Deve possuir rate limit próprio.

### Sincronização

Para prazo curto, um job Python no ambiente autorizado consulta Guests cuja foto
mudou, baixa cada foto uma vez pelo endpoint privado, gera o embedding com o
mesmo `buffalo_l` e grava por endpoint M2M de enrollment ou diretamente por uma
rotina controlada do backend. FTP não participa do reconhecimento.

Um endpoint incremental pode ser:

`GET /api/v1/access-control/face-sync?updatedAfter=<cursor>`

Ele retorna apenas `guestId`, estado de elegibilidade e uma versão/checksum da
foto, nunca a referência física. Remoção, mudança de status, Client inativo ou
troca de foto invalida imediatamente a entrada no índice.

## Persistência proposta

Tabela `guest_face_embeddings`:

- `guest_id` único e FK;
- `model_name` e `model_version`;
- `dimension`;
- `normalized`;
- `embedding` BLOB com 512 float32 em formato definido;
- `photo_sha256`;
- `status` (`ready`, `stale`, `failed`);
- `created_at` e `updated_at`.

O BLOB é biometria sensível: acesso restrito, backup protegido e retenção ligada
ao Guest. O índice em memória contém somente Guests elegíveis, mas a autorização
sempre consulta o estado atual no banco antes de retornar `allowed=true`.

`FACE_MATCH_THRESHOLD` deve ser uma configuração obrigatória e centralizada no
serviço facial. Nenhum valor de produção foi escolhido nesta entrega. A
calibração deve medir impostores e genuínos reais com o modelo exato e registrar
FAR/FRR. Embeddings normalizados usam produto escalar igual à similaridade de
cosseno.

## Autenticação de dispositivo

Criar credenciais próprias por dispositivo/serviço, armazenando somente hash no
backend. Campos mínimos: ID, nome, access point vinculado, hash da chave, scopes,
ativo, versão/rotação e timestamps. Comparar hashes em tempo constante, permitir
revogação e nunca registrar header ou chave.

O repositório continha uma chave com aparência real em uma versão anterior de
`.env.example`. Se ela já foi válida em qualquer ambiente, deve ser revogada e
substituída; removê-la da cópia atual não a remove do histórico Git.

Filtros separados devem distinguir `face:recognize` e `face:enroll`. Tokens
humanos do Portal não são aceitos nessas rotas.

## Eventos de acesso

Tabela `access_events`:

- `id`;
- `guest_id` nullable;
- `access_point`;
- `recognized`;
- `allowed`;
- `similarity` nullable;
- `reason`;
- `created_at`.

Recomenda-se também `device_id` e `event_id` único para auditoria/idempotência.
Não persistir frame, foto ou embedding consultado.

A consulta por frame não deve acionar porta nem produzir vários eventos finais.
Separar matching/decisão de um endpoint idempotente de confirmação de acesso, ou
implementar confirmação e cooldown no servidor. Liberação física permanece fora
do escopo desta entrega.

## Backend CI4 observado

O backend adjacente já possui `/api/v1`, envelope `{success,data}`, Guests,
Clients, fotos privadas e validação de Client ativo em fluxos humanos. Ainda não
possui device auth, endpoints faciais, embeddings, índice ou access events.

Arquivos de extensão previstos:

- `app/Config/Routes.php` e `app/Config/Filters.php`;
- `DeviceAuthFilter`, serviço/contexto e tabela de devices;
- `RecognitionController` e `RecognitionService`;
- `PrivateGuestPhotoService`;
- migrations/models de `guest_face_embeddings` e `access_events`;
- invalidação em `GuestRegistrationService` e `GuestService`.

Há uma inconsistência existente em `PhotoService`: a constante aponta para
`uploads/guests`, mas o código grava em `WRITEPATH/guests`. O backend deve
formalizar um único diretório antes de criar o endpoint privado.

## Teste controlado com Guest real

1. Implantar as migrations, autenticação M2M e endpoints em homologação.
2. Criar duas credenciais distintas: enrollment e dispositivo da porta.
3. Obter pelo endpoint privado somente a foto do Guest de teste completed.
4. Gerar e persistir o embedding com o mesmo modelo/artefato usado na porta.
5. Confirmar que a foto/versionamento consta como `ready` no índice.
6. Capturar a mesma pessoa ao vivo, sem terceiros, e registrar a similaridade.
7. Repetir em iluminação e distância controladas para calibrar o threshold.
8. Confirmar `allowed=true` somente com Guest completed e Client active.
9. Inativar o Client/Guest em homologação e confirmar negação imediata mesmo
   com embedding ainda presente.
10. Testar timeout, 401, 500 e índice vazio; todos devem negar.
11. Não conectar relé/fechadura durante esta etapa.

## Evoluções não incluídas

- Autorização offline com cache local criptografado e política de expiração.
- Liveness/anti-spoofing. A primeira versão recusa múltiplos rostos.
- Acionamento físico da porta.
- Deploy ou alteração de produção.
