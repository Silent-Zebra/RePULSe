import os
import glob

def list_query_pools(query_pool_dir='openrlhf/forecasting_rare_outputs/query_pools') -> list[str]:
    'Lists all .jsonl query pool files in the specified directory.'
    if not os.path.isdir(query_pool_dir):
        print(f'Query pool directory not found: {query_pool_dir}')
        return []
    
    pool_files = glob.glob(os.path.join(query_pool_dir, '*_queries.jsonl'))
    return [os.path.basename(f) for f in pool_files]

if __name__ == '__main__':
    print('Available query pools:')
    pools = list_query_pools()
    if pools:
        for pool_name in pools:
            print(f'- {pool_name}')
    else:
        print('No query pools found. Run query_generation.py to create them.') 