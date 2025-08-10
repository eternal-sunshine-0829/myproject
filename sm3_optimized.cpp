#include <iostream>
#include <cstring>
#include <iomanip>
#include <chrono>
using namespace std;
using namespace std::chrono;

typedef uint8_t  u8;
typedef uint32_t u32;
typedef uint64_t u64;

static inline u32 rotl32(u32 x, unsigned n) {
    unsigned m = n & 31u;     
    if (m == 0) return x;
    return (x << m) | (x >> (32 - m));
}

//�û�����
static inline u32 P0(u32 x) { return x ^ rotl32(x, 9) ^ rotl32(x, 17); }
static inline u32 P1(u32 x) { return x ^ rotl32(x, 15) ^ rotl32(x, 23); }

//��������FF��GG�������ִ�jѡ��ͬ�㷨
static inline u32 FFj(u32 x, u32 y, u32 z, int j) {
    return (j <= 15) ? (x ^ y ^ z) : ((x & y) | (x & z) | (y & z));
}
static inline u32 GGj(u32 x, u32 y, u32 z, int j) {
    return (j <= 15) ? (x ^ y ^ z) : ((x & y) | ((~x) & z));
}

//����Tj�������ִ�jѡ��ֵͬ
static inline u32 Tj_const(int j) {
    return (j <= 15) ? 0x79cc4519U : 0x7a879d8aU;
}

//��64�ֽڵĴ�������ݿ�ת��Ϊ16��u32��
void block_to_u32_be(const u8* block, u32 W16[16]) {
    for (int i = 0; i < 16; i++) {
        W16[i] = ((u32)block[4 * i] << 24) |
            ((u32)block[4 * i + 1] << 16) |
            ((u32)block[4 * i + 2] << 8) |
            ((u32)block[4 * i + 3]);
    }
}

//�Ż���ѹ����������ǰ������Ϣ��չ + 4��չ�� + Ԥ����Tj����
void sm3_compress(u32 V[8], const u8 block[64]) {
    u32 W[68];  
    u32 W1[64]; 

    //1. ��Ϣ��չ��ǰ16����ֱ�����������
    u32 W16[16];
    block_to_u32_be(block, W16);
    for (int i = 0; i < 16; ++i) W[i] = W16[i];

    //2. ����ʣ�����
    for (int j = 16; j < 68; ++j) {
        u32 tmp = W[j - 16] ^ W[j - 9] ^ rotl32(W[j - 3], 15);
        W[j] = P1(tmp) ^ rotl32(W[j - 13], 7) ^ W[j - 6];
    }

    //3. ����W1����
    for (int j = 0; j < 64; ++j) {
        W1[j] = W[j] ^ W[j + 4];
    }

    //4. Ԥ����(Tj <<< j)
    u32 T_rot[64];
    for (int j = 0; j < 64; ++j) {
        u32 t = Tj_const(j);
        T_rot[j] = rotl32(t, (unsigned)j);
    }

    //5. ��ʼ���Ĵ���A-H
    u32 A = V[0], B = V[1], C = V[2], D = V[3];
    u32 E = V[4], F = V[5], G = V[6], H = V[7];

    //6. 64��ѹ��
    for (int j = 0; j < 64; ++j) {
        u32 SS1 = rotl32((rotl32(A, 12) + E + T_rot[j]) & 0xFFFFFFFFU, 7);
        u32 SS2 = SS1 ^ rotl32(A, 12);
        u32 TT1 = (FFj(A, B, C, j) + D + SS2 + W1[j]) & 0xFFFFFFFFU;
        u32 TT2 = (GGj(E, F, G, j) + H + SS1 + W[j]) & 0xFFFFFFFFU;

        //���¼Ĵ���ֵ
        D = C;
        C = rotl32(B, 9);
        B = A;
        A = TT1;
        H = G;
        G = rotl32(F, 19);
        F = E;
        E = P0(TT2);
    }

    //7. ���¹�ϣֵ
    V[0] ^= A; V[1] ^= B; V[2] ^= C; V[3] ^= D;
    V[4] ^= E; V[5] ^= F; V[6] ^= G; V[7] ^= H;
}

//��Ϣ��亯��
size_t sm3_padding(const u8* data, size_t len, u8* out) {
    memcpy(out, data, len);
    out[len] = 0x80; // ��ӱ���'1'

    //������䳤��
    size_t pad_len = (len % 64 < 56) ? (56 - (len % 64)) : (120 - (len % 64));
    memset(out + len + 1, 0, pad_len - 1); // ���0

    //�����Ϣ���ȣ��������������
    u64 bit_len = (u64)len * 8;
    for (int i = 0; i < 8; ++i) {
        out[len + pad_len + i] = (u8)(bit_len >> (56 - 8 * i));
    }

    return len + pad_len + 8;
}

//SM3��ϣ������
void sm3_hash(const u8* data, size_t len, u8 digest[32]) {
    //��ʼ����ϣֵ
    u32 V[8] = {
        0x7380166FU, 0x4914B2B9U, 0x172442D7U, 0xDA8A0600U,
        0xA96F30BCU, 0x163138AAU, 0xE38DEE4DU, 0xB0FB0E4EU
    };

    //�����Ϣ
    u8 buf[128];
    size_t total_len = sm3_padding(data, len, buf);

    //����ÿ��512λ��
    for (size_t i = 0; i < total_len; i += 64) {
        sm3_compress(V, buf + i);
    }

    //����ϣֵת��Ϊ�ֽ����飨�����
    for (int i = 0; i < 8; ++i) {
        digest[4 * i] = (V[i] >> 24) & 0xFF;
        digest[4 * i + 1] = (V[i] >> 16) & 0xFF;
        digest[4 * i + 2] = (V[i] >> 8) & 0xFF;
        digest[4 * i + 3] = (V[i]) & 0xFF;
    }
}

int main() {
    const char* msg = "abc";
    u8 digest[32];

    sm3_hash((const u8*)msg, strlen(msg), digest);
    cout << "SM3(\"abc\") = ";
    for (int i = 0; i < 32; ++i)
        cout << hex << setw(2) << setfill('0') << (int)digest[i];
    cout << endl;

    //���ܲ���
    const int iterations = 100000;
    auto total_time = nanoseconds(0);
    for (int i = 0; i < iterations; ++i) {
        auto start = high_resolution_clock::now();
        sm3_hash((const u8*)msg, strlen(msg), digest);
        auto stop = high_resolution_clock::now();
        total_time += duration_cast<nanoseconds>(stop - start);
    }
    cout << dec << "Average time over " << iterations << " iterations (optimized SM3): "
        << total_time.count() / iterations << " ns" << endl;

    return 0;
}


