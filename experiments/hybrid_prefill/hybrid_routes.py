"""CPU-only routing contract: original BF16 tensors are complete only at context=0."""

def sequence_routes(starts, lengths):
    if len(starts)!=len(lengths)+1 or starts[0]!=0:
        raise ValueError('Invalid sequence boundaries')
    routes=[]
    for i,length in enumerate(lengths):
        start,end=starts[i:i+2]
        query_len=end-start
        if query_len<=0 or length<query_len:
            raise ValueError('Invalid query/context lengths')
        routes.append((start,end,length,query_len==length and query_len>1))
    return routes
