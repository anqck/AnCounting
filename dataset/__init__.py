# build dataset according to given 'dataset_file'
def build_dataset(args):
    if args.dataset_file == 'FSC147':

        from dataset.fsc147.loading_data import loading_data, collate_fn_fsc147
        return loading_data, collate_fn_fsc147

    return None

# def collate_fn_dataset(args):
#     if args.dataset_file == 'FSC147':
#         from dataset.fsc147.loading_data import collate_fn_dataset
#         return collate_fn_dataset

#     return None