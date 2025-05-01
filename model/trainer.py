import cupy as cp

class LinearRegressorGPU:
    def __init__(self, x_np, y_np, learning_rate=0.5, loss_function="mse", execution_mode="cupy"):
        self.x = cp.asarray(x_np)
        self.y = cp.asarray(y_np)
        self.learning_rate = learning_rate
        self.loss_function = loss_function.lower()
        self.execution_mode = execution_mode.lower()
        self.w = cp.array([0.0], dtype=cp.float32)
        self.b = cp.array([0.0], dtype=cp.float32)
        self.losses = []
        self.epoch = 0

        #----- only prepare kernels if using manual CUDA mode
        if self.execution_mode == "cuda":
            self._prepare_kernels()

    def _prepare_kernels(self):
        #----- Element-wise multiplication kernel (manual CUDA)
        self.multiply_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void multiply_kernel(const float* a, const float* b, float* out, int n) {
            int idx = blockDim.x * blockIdx.x + threadIdx.x;
            if (idx < n) {
                out[idx] = a[idx] * b[idx];
            }
        }
        ''', 'multiply_kernel')

    def compute_loss(self, error):
        if self.loss_function == "mse":
            return cp.mean(error ** 2)
        elif self.loss_function == "mae":
            return cp.mean(cp.abs(error))
        else:
            raise ValueError(f"Loss function '{self.loss_function}' not supported.")

    def step(self):
        y_pred = self.w * self.x + self.b
        error = y_pred - self.y

        if self.execution_mode == "cupy":
            grad_w = cp.mean(error * self.x)
            grad_b = cp.mean(error)
        elif self.execution_mode == "cuda":
            grad_w = self._manual_cuda_grad_w(error, self.x)
            grad_b = cp.mean(error)  #----- Simplified: still using cupy for grad_b

        self.w -= self.learning_rate * grad_w
        self.b -= self.learning_rate * grad_b

        loss = self.compute_loss(error).get()
        self.losses.append(loss)

        self.epoch += 1
        return loss

    def _manual_cuda_grad_w(self, error, x):
        n = error.size
        out = cp.empty_like(error)

        threads_per_block = 256
        blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

        self.multiply_kernel((blocks_per_grid,), (threads_per_block,), 
                             (error, x, out, n))

        grad_w = cp.mean(out)
        return grad_w

    def predict(self):
        return (self.w * self.x + self.b).get()

    def get_parameters(self):
        return self.w.get()[0], self.b.get()[0]
