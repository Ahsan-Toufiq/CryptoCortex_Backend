import time
import tenseal as ts  # type: ignore
from concrete.ml.torch.compile import compile_torch_model 

# -------------------------
# Helper functions for FHE processing
# -------------------------

def get_quantize_module(model_inf, torch_input, method="exact", bits=8):
    start = time.time()
    return compile_torch_model(
        model_inf,
        torch_input,
        n_bits=bits,
        rounding_threshold_bits={"n_bits": 6, "method": method}
    ), round(time.time() - start, 2)

def get_quantize_input(quantized_module, X_test_spikes):
    start = time.time()
    return quantized_module.quantize_input(X_test_spikes.detach().cpu().numpy()), round(time.time() - start, 2)


def get_encrypted_input(quantized_module, q_input):
    # Encrypt the input
    start = time.time()
    return quantized_module.fhe_circuit.encrypt(q_input[0:1, :, :]), round(time.time() - start, 2)

def get_encrypted_output(quantized_module, q_input_enc):
    # Execute the linear product in FHE
    start = time.time()
    return quantized_module.fhe_circuit.run(q_input_enc), round(time.time() - start, 2)

def get_decrypted_output(quantized_module, q_y_enc):
    # Decrypt the result (integer)
    start = time.time()
    return quantized_module.fhe_circuit.decrypt(q_y_enc), round(time.time() - start, 2)

def get_dequantize_output(quantized_module, q_y):
    # De-quantize and post-process the result
    start = time.time()
    return quantized_module.post_processing(quantized_module.dequantize_output(q_y)), round(time.time() - start, 2)

def get_normal_output(model_inf, X_test_spikes):
    start = time.time()
    return model_inf.forward(X_test_spikes), round(time.time() - start, 2)


def create_tenseal_context():
    context = ts.context(
        scheme=ts.SCHEME_TYPE.CKKS,  # Use CKKS for real numbers
        poly_modulus_degree=8192,  # Increase for better security but more overhead
        coeff_mod_bit_sizes=[60, 40, 40, 60],  # Security parameters
    )
    context.global_scale = 2**40
    context.generate_galois_keys()
    return context


def encrypt_with_tenseal(input_tensor, context):
    # Encrypt the input tensor using TenSEAL
    encrypted_tensor = ts.ckks_vector(context, input_tensor.flatten().tolist())
    return encrypted_tensor
