# Ingestion-to-tagger producer contract

The public EEGDash ingestion workflow and this tagger service are separate
components. Version 1.0.3 supplies an offline adapter, not an automatic scheduler
or a production deployment. The source catalogue keeps its BSD licence; this
service remains MIT.

## Prepare and inspect a request

After installing the service wheel, from any directory:

```sh
python -m src.ingestion dataset.json > enqueue-request.json
```

The input is one catalogue dataset document or an array. Required: nonempty
`dataset_id`, explicit HTTP(S) source URL (`external_links.github_url`,
`external_links.source_url` or top-level `source_url`), and supported evidence.
Embedded URL credentials are rejected. The adapter does not fetch source pages.

Source `name` maps to `title`; `description` maps to `dataset_description`.
Nonempty canonical fields win over aliases. `readme`, `participants_overview`,
`tasks`, `events`, `task_details` and `paper_abstract` pass through when supplied.
Text fields must be strings, tasks/events arrays of strings and task_details an
object. Participant counts do not establish health status. Abstract applicability
must be checked by the producer. Structured `tags`, `dxStatus`, `study_domain`
and prior model outputs are not copied. This is not a semantic scrubber for
label-like text embedded in raw documentation, which still requires source review.

The output matches `POST /api/v1/tag/enqueue/batch`. The adapter is also importable:

```python
from src.ingestion import build_enqueue_request
request = build_enqueue_request(dataset_document)
```

## Delivery boundary

Only deliver after confirming successful upstream ingestion and dataset identity.
Protect the service with deployment-level access control; never include secrets
in the snapshot. Persist the prepared request in the producer's durable outbox,
and mark it delivered only after a successful queue acknowledgement with the
expected total and queued/duplicate counts. Retry transport failures without
changing the snapshot; identical dataset/metadata jobs are deduplicated by the
existing PostgreSQL queue. No outbox/delivery daemon is provided here.

Queue acknowledgement is not completed tagging: inspect job status and retain
terminal failures for review. Once queued, worker model/write retries are bounded;
successful inference is cached for write retries. Changing only model/prompt
configuration does not automatically retag completed metadata; use the explicit
operator procedure in README.md. Machine-generated labels remain unreviewed.

## Evidence boundary

Tests verify field preservation through the installed library's model serializer,
invalid-input rejection, CLI export, and an opt-in disposable PostgreSQL/loopback
HTTP path through model failure, persistence failure and eventual cached write.
These synthetic checks do not establish classification accuracy or live source
ingestion reliability. The reviewed public EEGDash injector has not been modified
to call this adapter, and no production catalogue was retagged.
