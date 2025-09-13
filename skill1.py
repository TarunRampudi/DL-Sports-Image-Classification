import torch
import numpy as np
array =[[1,2,3],[4,5,6]]
print(array)
print("numy array type:{}",format(type(array)))
np_array = np.array(array)
print("numy array type:{}",format(type(np_array)))
print("numy array type:{}",format(np_array.shape))
print(np_array)
tensor = torch.Tensor(array)
print("PyTorch array type:{}",format(tensor.type))
print("PyTorch array type:{}",format(tensor.shape))
print(tensor)
np_ones =np.ones((2,5))
print(np_ones)
torch_ones =torch.ones((2,3))
print(torch_ones)
print(torch.arange(10))
print(torch.rand(3,4))
np_array = np.random.rand(2,3)
print(np_array)
tensor_from_np_array =torch.from_numpy(np_array)
print(tensor_from_np_array)
np_array_from_tensor = tensor_from_np_array.numpy()
print(np_array_from_tensor)
print(type(np_array_from_tensor))
print(np.allclose(np_array,np_array_from_tensor))
a= torch.rand(3,4)
b= torch.rand(3,4)
print(a)
print(b)

print(a.reshape(4,3))

print(a+b)
print(a-b)
print(a*b)
print(a/b)
print(b/a)

print(a @ b.T)

print(a.mean())
print(b.std())


print(a.mean().item())