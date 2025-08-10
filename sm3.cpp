#include <iostream>
#include <cstring>
#include <iomanip>
#include <chrono>
using namespace std;
using namespace std::chrono;

typedef uint8_t  u8;
typedef uint32_t u32;
typedef uint64_t u64;

//循环左移
u32 rotl32(u32 x, unsigned n) {
    n %= 32;
    return (x << n) | (x >> (32 - n));
}

//置换函数
u32 P0(u32 x) { return x ^ rotl32(x, 9) ^ rotl32(x, 17); }
u32 P1(u32 x) { return x ^ rotl32(x, 15) ^ rotl32(x, 23); }

//布尔函数FF和GG，根据轮次j选择不同算法
u32 FF(u32 x, u32 y, u32 z, int j) {
    return (j <= 15) ? (x ^ y ^ z) : ((x & y) | (x & z) | (y & z));
}

u32 GG(u32 x, u32 y, u32 z, int j) {
    return (j <= 15) ? (x ^ y ^ z) : ((x & y) | ((~x) & z));
}

//常量函数
u32 T(int j) {
    return (j <= 15) ? 0x79cc4519 : 0x7a879d8a;
}

//消息块转换（大端序）
void block_to_words(const u8* block, u32* words) {
    for (int i = 0; i < 16; i++) {
        words[i] = ((u32)block[4 * i] << 24) | ((u32)block[4 * i + 1] << 16) |
            ((u32)block[4 * i + 2] << 8) | ((u32)block[4 * i + 3]);
    }
}

//压缩函数
void compress(u32 state[8], const u8 block[64]) {
    u32 W[68], W1[64];

    //消息扩展
    block_to_words(block, W);
    for (int j = 16; j < 68; j++) {
        W[j] = P1(W[j - 16] ^ W[j - 9] ^ rotl32(W[j - 3], 15))
            ^ rotl32(W[j - 13], 7) ^ W[j - 6];
    }
    for (int j = 0; j < 64; j++) {
        W1[j] = W[j] ^ W[j + 4];
    }

    u32 A = state[0], B = state[1], C = state[2], D = state[3];
    u32 E = state[4], F = state[5], G = state[6], H = state[7];

    for (int j = 0; j < 64; j++) {
        u32 SS1 = rotl32(rotl32(A, 12) + E + rotl32(T(j), j), 7);
        u32 SS2 = SS1 ^ rotl32(A, 12);
        u32 TT1 = FF(A, B, C, j) + D + SS2 + W1[j];
        u32 TT2 = GG(E, F, G, j) + H + SS1 + W[j];

        D = C;
        C = rotl32(B, 9);
        B = A;
        A = TT1;
        H = G;
        G = rotl32(F, 19);
        F = E;
        E = P0(TT2);
    }

    //更新状态
    state[0] ^= A; state[1] ^= B; state[2] ^= C; state[3] ^= D;
    state[4] ^= E; state[5] ^= F; state[6] ^= G; state[7] ^= H;
}

//消息填充函数
size_t pad_message(const u8* input, size_t len, u8* output) {
    memcpy(output, input, len);
    output[len] = 0x80;

    size_t pad_len = (len % 64 < 56) ? 56 - len % 64 : 120 - len % 64;
    memset(output + len + 1, 0, pad_len - 1);

    u64 bit_len = len * 8;
    for (int i = 0; i < 8; i++) {
        output[len + pad_len + i] = (bit_len >> (56 - 8 * i)) & 0xff;
    }

    return len + pad_len + 8;
}

//哈希主函数
void sm3_hash(const u8* input, size_t len, u8 digest[32]) {
    //初始化哈希值
    u32 state[8] = {
        0x7380166f, 0x4914b2b9, 0x172442d7, 0xda8a0600,
        0xa96f30bc, 0x163138aa, 0xe38dee4d, 0xb0fb0e4e
    };

    //填充消息
    u8 buffer[128];
    size_t total_len = pad_message(input, len, buffer);

    //处理每个512位块
    for (size_t i = 0; i < total_len; i += 64) {
        compress(state, buffer + i);
    }

    //将哈希值转换为字节数组（大端序）
    for (int i = 0; i < 8; i++) {
        digest[4 * i] = (state[i] >> 24) & 0xff;
        digest[4 * i + 1] = (state[i] >> 16) & 0xff;
        digest[4 * i + 2] = (state[i] >> 8) & 0xff;
        digest[4 * i + 3] = state[i] & 0xff;
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

    //性能测试
    const int iterations = 100000;
    auto total_time = nanoseconds(0);
    for (int i = 0; i < iterations; ++i) {
        auto start = high_resolution_clock::now();
        sm3_hash((const u8*)msg, strlen(msg), digest);
        auto stop = high_resolution_clock::now();
        total_time += duration_cast<nanoseconds>(stop - start);
    }
    cout << dec << "Average time over " << iterations << " iterations: "
        << total_time.count() / iterations << " ns" << endl;

    return 0;
}
