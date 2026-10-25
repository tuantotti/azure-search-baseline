"""Local CWQ preparation and proxy relevance; independent of HTTP and Azure."""

import hashlib
import json
import random
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


def question_hash(text: str) -> str:
    normalized = ' '.join(unicodedata.normalize('NFKC', text).casefold().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


@dataclass(frozen=True)
class Record:
    id: str
    base_id: str
    question: str
    kind: str
    answers: frozenset[str]
    labels: tuple[str, ...]
    missing_labels: int = 0

    @property
    def key(self) -> str:
        return hashlib.sha256(self.id.encode()).hexdigest()

    @property
    def exclusion(self) -> str:
        base = self.base_id.replace("'", "''")
        return f"webqsp_id ne '{base}' and question_hash ne '{question_hash(self.question)}'"

    def document(self) -> dict:
        return {
            'chunk_id': self.key, 'document_id': self.id, 'webqsp_id': self.base_id,
            'question_hash': question_hash(self.question), 'composition_type': self.kind,
            'answer_ids': sorted(self.answers), 'title': self.question,
            'content': f"Question: {self.question}\nAnswers: {'; '.join(self.labels)}",
        }


@dataclass
class Dataset:
    training: list[Record]
    development: list[Record]
    hashes: dict[str, str]

    @classmethod
    def load(cls, directory: Path, train_limit: int = 0) -> 'Dataset':
        splits, hashes = {}, {}
        for split in ('train', 'dev'):
            raw = (directory / f'ComplexWebQuestions_{split}.json').read_bytes()
            hashes[split] = hashlib.sha256(raw).hexdigest()
            rows = json.loads(raw)
            records, seen = [], set()
            for row in rows:
                record_id, question = str(row['ID']), row['question'].strip()
                if record_id in seen or not question or not row.get('webqsp_ID'):
                    raise ValueError(f'Invalid or duplicate {split} record: {record_id}')
                seen.add(record_id)
                labels, ids, missing = [], set(), 0
                for answer in row['answers']:
                    if not answer.get('answer_id'):
                        raise ValueError(f'Missing answer ID: {record_id}')
                    ids.add(str(answer['answer_id']))
                    candidates = [answer.get('answer'), *(answer.get('aliases') or [])]
                    label = next((s.strip() for s in candidates if isinstance(s, str)
                                  and s.strip()), None)
                    if label:
                        labels.append(label)
                    else:
                        missing += 1
                records.append(Record(record_id, row['webqsp_ID'], question,
                                      row['compositionality_type'], frozenset(ids),
                                      tuple(dict.fromkeys(labels)), missing))
            splits[split] = records
        training = splits['train'][:train_limit] if train_limit else splits['train']
        if not training or not splits['dev']:
            raise ValueError('Both local dataset splits must be nonempty')
        return cls(training, splits['dev'], hashes)

    def sample(self, limit: int, seed: int) -> list[Record]:
        if limit < 0:
            raise ValueError('question limit must be nonnegative')
        count = min(limit or len(self.development), len(self.development))
        groups = defaultdict(list)
        for record in self.development:
            groups[record.kind].append(record)
        exact = {kind: len(rows) * count / len(self.development)
                 for kind, rows in groups.items()}
        quotas = {kind: int(value) for kind, value in exact.items()}
        remainder = count - sum(quotas.values())
        for kind in sorted(groups, key=lambda k: (-(exact[k] - quotas[k]), k))[:remainder]:
            quotas[kind] += 1
        rng, sampled = random.Random(seed), []
        for kind in sorted(groups):
            sampled.extend(rng.sample(groups[kind], quotas[kind]))
        rng.shuffle(sampled)
        return sampled

    def judgments(self, queries: list[Record]) -> dict[str, set[str]]:
        inverted = defaultdict(set)
        by_key = {r.key: r for r in self.training}
        for record in self.training:
            for answer in record.answers:
                inverted[answer].add(record.key)
        result = {}
        for query in queries:
            candidates = set().union(*(inverted[a] for a in query.answers))
            result[query.id] = {k for k in candidates
                                if by_key[k].base_id != query.base_id
                                and question_hash(by_key[k].question)
                                != question_hash(query.question)}
        return result
