import random
import hashlib
import time


class SM2:
    def __init__(self):
        self.p = 0xFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000FFFFFFFFFFFFFFFF
        self.a = 0xFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000FFFFFFFFFFFFFFFC
        self.b = 0x28E9FA9E9D9F5E344D5A9E4BCF6509A7F39789F515AB8F92DDBCBD414D940E93
        self.n = 0xFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFF7203DF6B21C6052B53BBF40939D54123  #基点的阶n
        self.Gx = 0x32C4AE2C1F1981195F9904466A39C9948FE30BBFF2660BE1715A4589334C74C7
        self.Gy = 0xBC3736A2F4F6779C59BDCEE36B692153D0A9877CC62A474002DF32E52139F0A0
        self.G = (self.Gx, self.Gy)  #基点 G

        #性能统计
        self.benchmark_results = {
            'keygen': [],
            'sign': [],
            'verify': []
        }

    #椭圆曲线加法运算
    def point_add(self, P, Q):
        if P == (0, 0):
            return Q
        if Q == (0, 0):
            return P
        if P[0] == Q[0] and P[1] != Q[1]:
            return (0, 0)

        if P == Q:
            #点加：同一点
            lam = (3 * P[0] * P[0] + self.a) * pow(2 * P[1], self.p - 2, self.p)
        else:
            #点加：不同点
            lam = (Q[1] - P[1]) * pow(Q[0] - P[0], self.p - 2, self.p)

        x3 = (lam * lam - P[0] - Q[0]) % self.p
        y3 = (lam * (P[0] - x3) - P[1]) % self.p

        return (x3, y3)

    #椭圆曲线标量乘法
    def point_mul(self, k, P):
        R = (0, 0)
        while k > 0:
            if k & 1:
                R = self.point_add(R, P)
            P = self.point_add(P, P)
            k >>= 1
        return R

    #生成SM2密钥对
    def generate_keypair(self):
        start_time = time.time()
        private_key = random.randint(1, self.n - 1)
        public_key = self.point_mul(private_key, self.G)
        elapsed = (time.time() - start_time) * 1000
        self.benchmark_results['keygen'].append(elapsed)
        return private_key, public_key

    #SM2签名算法
    def sm2_sign(self, private_key, message, ZA=None):
        start_time = time.time()
        if ZA is None:
            ZA = b'\x00' * 32

        #e = Hash(ZA || M)
        e = hashlib.sha256(ZA + message).digest()
        e = int.from_bytes(e, 'big')

        while True:
            k = random.randint(1, self.n - 1)
            x1, y1 = self.point_mul(k, self.G)
            r = (e + x1) % self.n
            if r == 0 or r + k == self.n:
                continue

            s = (pow(private_key + 1, self.n - 2, self.n) * (k - r * private_key)) % self.n
            if s == 0:
                continue

            elapsed = (time.time() - start_time) * 1000
            self.benchmark_results['sign'].append(elapsed)
            return (r, s)

    #SM2验证算法
    def sm2_verify(self, public_key, message, signature, ZA=None):
        start_time = time.time()

        if ZA is None:
            ZA = b'\x00' * 32

        r, s = signature
        if not (1 <= r < self.n and 1 <= s < self.n):
            return False

        e = hashlib.sha256(ZA + message).digest()
        e = int.from_bytes(e, 'big')

        t = (r + s) % self.n
        if t == 0:
            return False

        x1, y1 = self.point_add(self.point_mul(s, self.G), self.point_mul(t, public_key))
        R = (e + x1) % self.n

        elapsed = (time.time() - start_time) * 1000
        self.benchmark_results['verify'].append(elapsed)

        return R == r

    def print_benchmark_results(self, iterations=1):
        print(f"\n=== 性能测试结果 (运行 {iterations} 次) ===")
        for key, values in self.benchmark_results.items():
            if values:
                avg = sum(values) / len(values)
                print(f"{key} 平均耗时: {avg:.3f} ms")


if __name__ == "__main__":
    sm2 = SM2()
    iterations = 10  #测试次数

    for i in range(iterations):
        #生成密钥对
        private_key, public_key = sm2.generate_keypair()
        if i == 0:
            print(f"Private key: {hex(private_key)}")
            print(f"Public key: ({hex(public_key[0])}, {hex(public_key[1])})")

        #签名与验证
        message = b"Hello, World!"
        signature = sm2.sm2_sign(private_key, message)
        if i == 0:
            print(f"Signature: (r={hex(signature[0])}, s={hex(signature[1])})")

        is_valid = sm2.sm2_verify(public_key, message, signature)
        if i == 0:
            print(f"Signature valid: {is_valid}")

    sm2.print_benchmark_results(iterations)
