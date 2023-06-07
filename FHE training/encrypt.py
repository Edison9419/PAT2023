import seal
from seal import *


def encrypt_init():
    print(seal.__version__)

    parms = EncryptionParameters(scheme_type.bgv)
    poly_modulus_degree = 8192
    parms.set_poly_modulus_degree(poly_modulus_degree)
    parms.set_coeff_modulus(CoeffModulus.BFVDefault(poly_modulus_degree))
    parms.set_plain_modulus(PlainModulus.Batching(poly_modulus_degree, 20))
    context = SEALContext(parms)

    keygen = KeyGenerator(context)
    secret_key = keygen.secret_key()
    public_key = keygen.create_public_key()
    relin_keys = keygen.create_relin_keys()

    encryptor = Encryptor(context, public_key)
    evaluator = Evaluator(context)
    decryptor = Decryptor(context, secret_key)

    return encryptor, evaluator, decryptor


def encrypt_cal(encryptor, evaluator, decryptor, p, g, rate, offset):
    p_scaled = int((p + offset) * rate)
    g_scaled = int((g + offset) * rate)

    plaintext_param = Plaintext(str(p_scaled))
    ciphertext_param = Ciphertext()
    ciphertext_param = encryptor.encrypt(plaintext_param)

    plaintext_grad = Plaintext(str(g_scaled))
    ciphertext_grad = Ciphertext()
    ciphertext_grad = encryptor.encrypt(plaintext_grad)

    evaluator.add_inplace(ciphertext_param, ciphertext_grad)

    updated_param = Plaintext()
    updated_param = decryptor.decrypt(ciphertext_param)

    return float(int(updated_param.to_string(), 16) / rate - offset)