"""Select three interventions from a *premeasured*, shared fixed-parent panel."""
from anyrsi_runtime.group import BestOfN, Member, MergePositive


def pick(parent, whole, atoms, step=2):
    """Only complete paired task vectors are accepted. Inputs are selection scores only."""
    if not parent or len(whole) != len(atoms) or not whole:
        raise ValueError('need equal, non-empty discovery slots per policy')
    if any(set(s) != set(parent) for s in [*whole, *atoms]):
        raise ValueError('different tasks across selectable candidates')

    def choice(scores):
        members = [Member(i, s) for i, s in enumerate(scores)]
        d = BestOfN().decide(parent, members, 0.05)
        return d.chosen if d.chosen is not None and d.advantage[d.chosen] > 0 else None

    merged = MergePositive(step=step).decide(parent, [Member(i, s) for i, s in enumerate(atoms)], 0.05)
    return {'whole': choice(whole), 'best_atom': choice(atoms), 'merge': merged.merge or []}
