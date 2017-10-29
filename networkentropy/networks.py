from networkentropy import __networks__


def get_network_info(network_name):
    network_data = __networks__[network_name]
    return network_data

def load_python(df):
    g = nx.DiGraph()
    g.add_nodes_from(df.package_name.unique())
    edges = df.loc[df.requirement.notnull(), ['package_name', 'requirement']].values
    g.add_edges_from(edges)

    g.remove_nodes_from(['.', 'nan', np.nan])

    deg = g.degree()
    to_remove = [n for n in deg if deg[n] <= 0]
    g.remove_nodes_from(to_remove)
    return g

def load_network(network_name):

    network_data = __networks__[network_name]

    this_dir, this_filename = os.path.split(__file__)
    file = os.path.join(this_dir, 'data', network_data['file'])

    df = pd.read_csv(file)

    if network_name == 'python':
        graph = load_python(df)

    return graph


