"""Three measurements of one parent, then three arms at the same search HTTP ceiling."""
import argparse
import asyncio
import json
import os
from pathlib import Path

from anyrsi import Item
from anyrsi.types import for_agent
from anyrsi_judge import EvidenceJudge
from anyrsi_llm import ChatAgent, request_observer
from loop_llm import BASE, HardOpsEnv, run_arm
from request_budget import RequestBudget


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


async def experiment(args):
    args.root.mkdir(parents=True, exist_ok=False)
    save(args.root / 'protocol.json', {**vars(args), 'root': str(args.root), 'split': 'disjoint-v2',
         'design': 'same parent at initialization; subsequently each arm inherits its admitted edits',
         'budget': 'per-arm search HTTP attempts, including solver, proposer and retries; final separate'})
    for seed in args.seeds:
        env = HardOpsEnv(seed=seed, easy=not args.hard)
        tasks = env.tasks('holdout')
        if not tasks:
            raise ValueError("missing noise tasks")
        agent = ChatAgent('https://api.openai.com/v1', args.model, temperature=None,
                          max_tokens=4000, api_key_env='OPENAI_API_KEY')
        judge = EvidenceJudge(rubric={'items': [{'id': 'correct', 'check': 'output_equals', 'required': True,
                              'params': {'extract': r'=\s*(-?\d+)\s*$'}}]})
        rows = []
        meter = RequestBudget(None, args.root / f'noise-{seed}-requests.jsonl')
        mark = request_observer.set(meter)
        try:
            with meter.phase('noise'):
                for _ in range(3):
                    scores = {}
                    for task in tasks:
                        visible = for_agent(task)
                        trace = agent.run(visible, [Item(id='instructions', kind='prompt', text=BASE)], 'parent')
                        if trace.metadata.get('error') or trace.output is None:
                            raise RuntimeError(f'failed model call for {task.id}; not a solver zero')
                        evidence = judge.collect(task, trace)
                        scores[task.id] = judge.evaluate(task, trace, evidence).score
                    rows.append(scores)
                    save(args.root / f'noise-{seed}.json', {'pool_hash': env.pool.digest, 'rows': rows,
                         'flips': [k for k in scores if len({r[k] for r in rows}) > 1], 'usage': meter.report()})
        finally:
            request_observer.reset(mark)

        for arm in ['bon:patch', 'atom:patch', 'grad:patch']:
            tag = f'{seed}-{arm.replace(":", "_")}'
            save(args.root / 'progress.json', {'stage': 'search', 'seed': seed, 'arm': arm})
            result = await run_arm(arm, args.root / tag, seed, args.model, 'OPENAI_API_KEY',
                                   1000, args.size, not args.hard, 0, args.requests, args.step)
            save(args.root / (tag + '.json'), result)
            if result['budget']['search_used'] != args.requests or result['pool_hash'] != env.pool.digest:
                raise RuntimeError('unequal search use or different task partition')
    save(args.root / 'progress.json', {'stage': 'completed'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--model', default='gpt-6.1-terra')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--requests', type=int, default=360)
    parser.add_argument('--size', type=int, default=2)
    parser.add_argument('--step', type=int, default=2)
    parser.add_argument('--hard', action='store_true')
    asyncio.run(experiment(parser.parse_args()))
