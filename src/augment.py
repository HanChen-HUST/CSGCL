from cdlib import algorithms


def get_cd_algorithm(name):
    algs = {
        # non-overlapping algorithms
        'louvain': algorithms.louvain,
        'combo': algorithms.pycombo,
        'leiden': algorithms.leiden,
        'ilouvain': algorithms.ilouvain,
        'edmot': algorithms.edmot,
        'eigenvector': algorithms.eigenvector,
        'girvan_newman': algorithms.girvan_newman,
        # overlapping algorithms
        'demon': algorithms.demon,
        'lemon': algorithms.lemon,
        'ego-splitting': algorithms.egonet_splitter,
        'nnsed': algorithms.nnsed,
        'lpanni': algorithms.lpanni,
    }
    return algs[name]
