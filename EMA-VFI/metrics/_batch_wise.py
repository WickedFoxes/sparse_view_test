def batch_wise(func):
    def wrapper(tensor_a, tensor_b):
        assert tensor_a.shape == tensor_b.shape, (f'Image shapes are different: {tensor_a.shape}, {tensor_b.shape}.')
        assert len(tensor_a.shape) == 4, 'The input should be 4D tensor'
        assert tensor_a.min() >= 0 and tensor_a.max() <= 1, 'The value of tensor_a should be in [0, 1]'
        assert tensor_b.min() >= 0 and tensor_b.max() <= 1, 'The value of tensor_b should be in [0, 1]'
        for i in range(tensor_a.shape[0]):
            yield func(tensor_a[i], tensor_b[i])
    return wrapper

