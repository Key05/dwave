from dwave.cloud import Client

with Client.from_config() as client:
    solver = client.get_solver(qpu=True)
    print('solver: ' + solver.data['id'])

    u, v = next(iter(solver.edges))
    print('edges: ' + str([u, v]))
    Q = { 
        (u, u): -3, 
        (u, v): 4,
        (v, u): 0,
        (v, v): -4, 
    }   

    computation = solver.sample_qubo(Q, num_reads=5)

    for sample in computation.samples:
        print('result: ' + str([int(sample[u]), int(sample[v])]))

    print('timing: ' + str(computation.timing))
