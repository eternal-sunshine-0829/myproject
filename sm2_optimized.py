import secrets
import hashlib
import time

class OptimizedSM2WithBenchmark:
    def __init__(self):
        self.p = 0xFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000FFFFFFFFFFFFFFFF
        self.a = 0xFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000FFFFFFFFFFFFFFFC
        self.b = 0x28E9FA9E9D9F5E344D5A9E4BCF6509A7F39789F515AB8F92DDBCBD414D940E93
        self.n = 0xFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFF7203DF6B21C6052B53BBF40939D54123
        self.Gx = 0x32C4AE2C1F1981195F9904466A39C9948FE30BBFF2660BE1715A4589334C74C7
        self.Gy = 0xBC3736A2F4F6779C59BDCEE36B692153D0A9877CC62A474002DF32E52139F0A0
        self.G = (self.Gx, self.Gy)

        #预计算表
        self.precompute_table = self._build_precompute_table()

        #性能统计
        self.benchmark_results = {
            'keygen': [],
            'sign': [],
            'verify': [],
        }

    #仿射 -> 雅可比坐标
    def _affine_to_jacobian(self, P):
        if P == (0, 0):
            return (0, 0, 0)
        return (P[0], P[1], 1)

    #雅可比 -> 仿射坐标
    def _jacobian_to_affine(self, P):
        X, Y, Z = P
        if Z == 0:
            return (0, 0)
        #Z^{-1} mod p
        z_inv = pow(Z, self.p - 2, self.p)
        z_inv_sq = (z_inv * z_inv) % self.p
        x = (X * z_inv_sq) % self.p
        y = (Y * z_inv_sq * z_inv) % self.p
        return (x, y)

    #雅可比坐标下点加倍
    def _point_double_jacobian(self, P):
        X, Y, Z = P
        if Z == 0:
            return (0, 0, 0)


        Y_sq = (Y * Y) % self.p
        S = (4 * X * Y_sq) % self.p
        Z_sq = (Z * Z) % self.p
        Z_4 = (Z_sq * Z_sq) % self.p
        M = (3 * (X * X % self.p) + (self.a * Z_4) % self.p) % self.p

        X_new = (M * M - 2 * S) % self.p
        Y_new = (M * (S - X_new) - 8 * (Y_sq * Y_sq % self.p)) % self.p
        Z_new = (2 * Y * Z) % self.p

        return (X_new, Y_new, Z_new)

    #雅可比坐标下点加
    def _point_add_jacobian(self, P, Q):
        X1, Y1, Z1 = P
        X2, Y2, Z2 = Q

        if Z1 == 0:
            return (X2, Y2, Z2)
        if Z2 == 0:
            return (X1, Y1, Z1)

        Z1_sq = (Z1 * Z1) % self.p
        Z2_sq = (Z2 * Z2) % self.p

        U1 = (X1 * Z2_sq) % self.p
        U2 = (X2 * Z1_sq) % self.p
        S1 = (Y1 * Z2 * Z2_sq) % self.p
        S2 = (Y2 * Z1 * Z1_sq) % self.p

        if U1 == U2:
            if S1 != S2:
                #P + (-P) = O
                return (0, 0, 0)
            #P == Q
            return self._point_double_jacobian(P)

        H = (U2 - U1) % self.p
        R = (S2 - S1) % self.p
        H_sq = (H * H) % self.p
        H_cu = (H * H_sq) % self.p

        X3 = (R * R - H_cu - 2 * U1 * H_sq) % self.p
        Y3 = (R * (U1 * H_sq - X3) - S1 * H_cu) % self.p
        Z3 = (H * Z1 * Z2) % self.p

        return (X3, Y3, Z3)

    #计算k的NAF（非相邻形式）表示
    def _naf(self, k):
        naf = []
        while k > 0:
            if k & 1:
                ki = 2 - (k % 4)  # 生成 ±1
                k = k - ki
                naf.append(ki)
            else:
                naf.append(0)
            k >>= 1
        return naf

    #基于 NAF 的点乘
    def _point_mul_naf(self, k, P):
        #start_time = time.perf_counter()

        naf = self._naf(k)
        Q = (0, 0, 0)
        P_jac = self._affine_to_jacobian(P)

        #从高位到低位
        for digit in reversed(naf):
            Q = self._point_double_jacobian(Q)
            if digit == 1:
                Q = self._point_add_jacobian(Q, P_jac)
            elif digit == -1:
                #雅可比下的取负
                negP = (P_jac[0], (-P_jac[1]) % self.p, P_jac[2])
                Q = self._point_add_jacobian(Q, negP)

        #elapsed = (time.perf_counter() - start_time) * 1000.0
        #self.benchmark_results['naf_point_mul'].append(elapsed)

        return self._jacobian_to_affine(Q)

    #基点G的预计算表
    def _build_precompute_table(self):
        #对于每个 i(0..bitlen-1)，存储 2^i * G 的仿射坐标
        table = {}
        bitlen = self.n.bit_length()
        P_jac = self._affine_to_jacobian(self.G)
        for i in range(bitlen):
            #将当前雅可比点转换为仿射并存储（若为无穷点则为 (0,0)）
            table[i] = self._jacobian_to_affine(P_jac)
            #在雅可比下做点加倍以得到下一个 2^{i+1} * G
            P_jac = self._point_double_jacobian(P_jac)
        return table

    #对基点 G 做固定基点乘：使用预计算表计算 k*G
    def _fixed_point_mul(self, k):
        #start_time = time.perf_counter()

        result = (0, 0, 0)
        bitlen = len(self.precompute_table)
        #遍历低位到高位，若该位为 1 则把对应预计算的 2^i * G 加到结果
        for i in range(bitlen):
            if (k >> i) & 1:
                pre_aff = self.precompute_table[i]
                pre_jac = self._affine_to_jacobian(pre_aff)
                result = self._point_add_jacobian(result, pre_jac)

        #elapsed = (time.perf_counter() - start_time) * 1000.0
        #self.benchmark_results['fixed_point_mul'].append(elapsed)

        return self._jacobian_to_affine(result)

    #生成 SM2 密钥对
    def generate_keypair(self):
        start_time = time.perf_counter()
        private_key = secrets.randbelow(self.n - 1) + 1
        public_key = self._fixed_point_mul(private_key)

        elapsed = (time.perf_counter() - start_time) * 1000.0
        self.benchmark_results['keygen'].append(elapsed)

        return private_key, public_key

    def sm2_sign(self, private_key, message, ZA=None):
        start_time = time.perf_counter()
        if ZA is None:
            ZA = b'\x00' * 32

        e_bytes = hashlib.sha256(ZA + message).digest()
        e = int.from_bytes(e_bytes, 'big')

        while True:
            k = secrets.randbelow(self.n - 1) + 1
            x1, y1 = self._fixed_point_mul(k)
            r = (e + x1) % self.n
            if r == 0 or r + k == self.n:
                continue

            # s = ((1 + d)^-1 * (k - r*d)) mod n
            inv = pow(private_key + 1, -1, self.n)
            s = (inv * (k - r * private_key)) % self.n
            if s == 0:
                continue

            elapsed = (time.perf_counter() - start_time) * 1000.0
            self.benchmark_results['sign'].append(elapsed)

            return (r, s)

    def sm2_verify(self, public_key, message, signature, ZA=None):
        start_time = time.perf_counter()
        if ZA is None:
            ZA = b'\x00' * 32

        r, s = signature
        #基本范围检验
        if not (1 <= r < self.n and 1 <= s < self.n):
            return False

        e_bytes = hashlib.sha256(ZA + message).digest()
        e = int.from_bytes(e_bytes, 'big')

        t = (r + s) % self.n
        if t == 0:
            return False

        #计算 s·G + t·P
        sG = self._point_mul_naf(s, self.G)
        tP = self._point_mul_naf(t, public_key)

        #将仿射转换为雅可比并相加，再转回仿射
        if sG == (0, 0):
            sum_aff = tP
        elif tP == (0, 0):
            sum_aff = sG
        else:
            sum_jac = self._point_add_jacobian(
                self._affine_to_jacobian(sG),
                self._affine_to_jacobian(tP)
            )
            sum_aff = self._jacobian_to_affine(sum_jac)

        x1, _ = sum_aff
        R = (e + x1) % self.n
        elapsed = (time.perf_counter() - start_time) * 1000.0
        self.benchmark_results['verify'].append(elapsed)
        return R == r

    def print_benchmark_results(self, iterations=1):
        print(f"\n=== 性能测试结果 (平均{iterations}次运行) ===")
        for key in self.benchmark_results:
            if self.benchmark_results[key]:
                avg = sum(self.benchmark_results[key]) / len(self.benchmark_results[key])
                print(f"{key}: {avg:.3f} ms")


if __name__ == "__main__":
    sm2 = OptimizedSM2WithBenchmark()
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
