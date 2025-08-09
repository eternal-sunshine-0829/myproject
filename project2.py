import cv2
import numpy as np
import os
from math import ceil

#嵌入区域
REGIONS = [
    #低频区域
    (10, 100, 10, 100),  #左上低频区 90x90
    (10, 100, 412, 502),  #右上低频区 90x90
    (412, 502, 10, 100),  #左下低频区 90x90
    (412, 502, 412, 502),  #右下低频区 90x90
    #中频区域
    (150, 250, 150, 250),  #中心区域 100x100
]

def str_to_bitstring(s):
    return ''.join(f'{ord(c):08b}' for c in s)

#汉明码 (7,4)
def hamming74_encode(bits):
    encoded = []
    for i in range(0, len(bits), 4):
        d = [int(b) for b in bits[i:i + 4].ljust(4, '0')]
        p1 = d[0] ^ d[1] ^ d[3]
        p2 = d[0] ^ d[2] ^ d[3]
        p3 = d[1] ^ d[2] ^ d[3]
        encoded += [p1, p2, d[0], p3, d[1], d[2], d[3]]
    return encoded

def hamming74_decode(bits):
    decoded = []
    for i in range(0, len(bits), 7):
        block = bits[i:i + 7]
        if len(block) < 7:
            break
        b = block.copy()
        p1, p2, d0, p3, d1, d2, d3 = b
        s1 = p1 ^ d0 ^ d1 ^ d3
        s2 = p2 ^ d0 ^ d2 ^ d3
        s3 = p3 ^ d1 ^ d2 ^ d3
        error_pos = s1 * 1 + s2 * 2 + s3 * 4
        if error_pos != 0 and error_pos <= 7:
            b[error_pos - 1] ^= 1
        decoded += [b[2], b[4], b[5], b[6]]
    return decoded


def bitstring_to_str(bits):
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i + 8]
        if len(byte) < 8:
            break
        chars.append(chr(int(''.join(str(b) for b in byte), 2)))
    return ''.join(chars)


#Zig-zag索引
def zigzag_indices(n=8):
    #返回8x8的zigzag索引
    idxs = []
    for s in range(2 * n - 1):
        if s % 2 == 0:
            for i in range(s + 1):
                j = s - i
                if i < n and j < n:
                    idxs.append((i, j))
        else:
            for i in range(s, -1, -1):
                j = s - i
                if i < n and j < n:
                    idxs.append((i, j))
    return idxs

ZZ = zigzag_indices(8)


#将图像按8x8块做DCT/IDCT
def block_process(channel_float, block_size=8, func=None):
    h, w = channel_float.shape
    out = np.zeros_like(channel_float)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = channel_float[i:i + block_size, j:j + block_size]
            if block.shape[0] != block_size or block.shape[1] != block_size:
                #填充
                padded = np.zeros((block_size, block_size), dtype=block.dtype)
                padded[:block.shape[0], :block.shape[1]] = block
                block = padded
            D = cv2.dct(block)
            if func is not None:
                D = func(D)
            idct_block = cv2.idct(D)
            out[i:i + block_size, j:j + block_size] = idct_block[:block.shape[0], :block.shape[1]]
    return out


#全局低频系数位置生成
def get_lowfreq_positions(h, w, block_size=8, coeffs_per_block=6, skip_dc=True):
    positions = []
    blocks_i = h // block_size
    blocks_j = w // block_size
    for bi in range(blocks_i):
        for bj in range(blocks_j):
            #对每个块，从zigzag序列中选取前coeffs_per_block个系数
            count = 0
            for (u, v) in ZZ:
                if skip_dc and (u == 0 and v == 0):
                    continue
                positions.append((bi, bj, u, v))
                count += 1
                if count >= coeffs_per_block:
                    break
    return positions


#特征对齐（ORB + RANSAC 单应矩阵）
def align_image_by_features(target_img, template_img):
    #target_img: 待校正图
    #template_img: 原始参考图
    #返回warp后的图像（与template_img对齐），若失败则返回原始target_img
    try:
        orb = cv2.ORB_create(2000)
        kp1, des1 = orb.detectAndCompute(cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY), None)
        kp2, des2 = orb.detectAndCompute(cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY), None)
        if des1 is None or des2 is None:
            return target_img  #无法匹配
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        if len(matches) < 10:
            return target_img
        #取前N个匹配
        good = matches[:min(200, len(matches))]
        ptsA = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        ptsB = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(ptsB, ptsA, cv2.RANSAC, 5.0)
        if H is None:
            return target_img
        h, w = template_img.shape[:2]
        aligned = cv2.warpPerspective(target_img, H, (w, h), borderValue=(128, 128, 128))
        return aligned
    except Exception as e:
        #匹配失败则返回原图
        return target_img


#嵌入水印
def embed_watermark(input_image_path, watermark_text, output_image_path,
                    alpha=0.25, repeat=5, coeffs_per_block=6):
    img = cv2.imread(input_image_path)
    img = cv2.resize(img, (512, 512))

    #转到YCrCb，主要在Y通道嵌入（亮度更鲁棒）
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycc)
    Y_float = np.float32(Y) / 255.0

    #将水印文本转换为比特序列
    watermark_bits = ''.join(format(ord(c), '08b') for c in watermark_text)
    encoded_bits = hamming74_encode(watermark_bits)
    #扩展冗余：每个编码比特重复repeat次并在多处嵌入
    bits_repeated = [b for b in encoded_bits for _ in range(repeat)]

    #全局低频位置（按8x8块）
    h, w = Y.shape
    positions = get_lowfreq_positions(h, w, block_size=8, coeffs_per_block=coeffs_per_block, skip_dc=True)
    #把bits填入positions的前len(bits_repeated)个位置（若不够则循环嵌入）
    total_positions = len(positions)
    if total_positions == 0:
        raise RuntimeError("没有可用位置进行嵌入")
    #将Y拆成8x8 block的DCT矩阵集合（在内存中修改）
    blocks_per_row = w // 8
    blocks_per_col = h // 8
    D = np.zeros_like(Y_float)
    #先计算每个block的DCT
    for bi in range(blocks_per_col):
        for bj in range(blocks_per_row):
            i0, j0 = bi * 8, bj * 8
            block = Y_float[i0:i0 + 8, j0:j0 + 8]
            if block.shape[0] != 8 or block.shape[1] != 8:
                continue
            D[i0:i0 + 8, j0:j0 + 8] = cv2.dct(block)

    #把bits写入低频系数
    idx = 0
    #将bits分散到多个块中以增加冗余
    for rep_pass in range(ceil(len(bits_repeated) / total_positions)):
        for p_i, pos in enumerate(positions):
            if idx >= len(bits_repeated):
                break
            bi, bj, u, v = pos
            i0, j0 = bi * 8, bj * 8
            #边界检查
            if i0 + u >= h or j0 + v >= w:
                continue
            #将比特嵌入系数：通过微幅加/减alpha
            bit = bits_repeated[idx]
            #在系数上叠加一个可被检测的偏移量
            if bit == 1:
                D[i0 + u, j0 + v] += alpha
            else:
                D[i0 + u, j0 + v] -= alpha
            idx += 1
        if idx >= len(bits_repeated):
            break

    #逆变换回空间域
    Y_mod = np.zeros_like(Y_float)
    for bi in range(blocks_per_col):
        for bj in range(blocks_per_row):
            i0, j0 = bi * 8, bj * 8
            blockD = D[i0:i0 + 8, j0:j0 + 8]
            if blockD.shape[0] != 8 or blockD.shape[1] != 8:
                continue
            idct_block = cv2.idct(blockD)
            Y_mod[i0:i0 + 8, j0:j0 + 8] = idct_block

    Y_mod = np.uint8(np.clip(Y_mod * 255.0, 0, 255))
    ycc_mod = cv2.merge([Y_mod, Cr, Cb])
    img_mod = cv2.cvtColor(ycc_mod, cv2.COLOR_YCrCb2BGR)

    cv2.imwrite(output_image_path, img_mod)
    print(f"水印嵌入完成，保存到: {output_image_path}")
    return True


#从Y通道全局低频提取水印
def extract_watermark(watermarked_path, original_path, watermark_text,
                      alpha=0.25, repeat=5, coeffs_per_block=6):
    img_wm = cv2.imread(watermarked_path)
    img_orig = cv2.imread(original_path)
    if img_wm is None or img_orig is None:
        raise FileNotFoundError("找不到输入图片")

    img_wm = cv2.resize(img_wm, (512, 512))
    img_orig = cv2.resize(img_orig, (512, 512))

    #先对齐：把受攻击图像对齐到原图
    img_wm_aligned = align_image_by_features(img_wm, img_orig)

    #转为YCrCb，只在Y通道提取
    ycc_wm = cv2.cvtColor(img_wm_aligned, cv2.COLOR_BGR2YCrCb)
    ycc_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2YCrCb)
    Y_wm = np.float32(ycc_wm[:, :, 0]) / 255.0
    Y_orig = np.float32(ycc_orig[:, :, 0]) / 255.0

    h, w = Y_wm.shape
    positions = get_lowfreq_positions(h, w, block_size=8, coeffs_per_block=coeffs_per_block, skip_dc=True)
    total_positions = len(positions)
    if total_positions == 0:
        raise RuntimeError("没有可用于提取的位置")

    #计算DCT集合
    D_wm = np.zeros_like(Y_wm)
    D_orig = np.zeros_like(Y_orig)
    blocks_per_row = w // 8
    blocks_per_col = h // 8
    for bi in range(blocks_per_col):
        for bj in range(blocks_per_row):
            i0, j0 = bi * 8, bj * 8
            block_wm = Y_wm[i0:i0 + 8, j0:j0 + 8]
            block_orig = Y_orig[i0:i0 + 8, j0:j0 + 8]
            if block_wm.shape[0] != 8 or block_wm.shape[1] != 8:
                continue
            D_wm[i0:i0 + 8, j0:j0 + 8] = cv2.dct(block_wm)
            D_orig[i0:i0 + 8, j0:j0 + 8] = cv2.dct(block_orig)

    #读取嵌入时的编码比特长度
    original_bits = ''.join(format(ord(c), '08b') for c in watermark_text)
    encoded_bits = hamming74_encode(original_bits)
    L_enc = len(encoded_bits)

    #重复检测并收集votes
    detected_bits_votes = [[] for _ in range(L_enc)]

    #从positions中按顺序读取并反向判断比特（按repeat重复合并）
    #我们知道嵌入时是把encoded_bits展开并重复repeat次，然后循环分布到positions，所以读取过程要模拟这个过程并对每个编码位做投票
    #先把positions分成passes，计算可用的embed slots = total_positions * passes >= L_enc * repeat
    #选择合适的passes
    passes = ceil((L_enc * repeat) / total_positions)
    idx = 0
    for rep_pass in range(passes):
        for pos in positions:
            if idx >= L_enc * repeat:
                break
            bi, bj, u, v = pos
            i0, j0 = bi * 8, bj * 8
            #边界检查
            if i0 + u >= h or j0 + v >= w:
                idx += 1
                continue
            diff = (D_wm[i0 + u, j0 + v] - D_orig[i0 + u, j0 + v]) / alpha
            bit_detected = 1 if diff > 0 else 0
            enc_pos = (idx % (L_enc * repeat)) // repeat
            detected_bits_votes[enc_pos].append(bit_detected)
            idx += 1
        if idx >= L_enc * repeat:
            break

    final_encoded = []
    for i in range(L_enc):
        votes = detected_bits_votes[i]
        if len(votes) == 0:
            final_encoded.append(0)
        else:
            final_encoded.append(1 if sum(votes) > len(votes) / 2 else 0)

    #纠错解码
    corrected_bits = hamming74_decode(final_encoded)
    watermark = bitstring_to_str(corrected_bits)

    #对比准确率
    original_decoded = hamming74_decode(encoded_bits)
    #防止长度不一致
    minlen = min(len(original_decoded), len(corrected_bits))
    if minlen == 0:
        bit_accuracy = 0.0
    else:
        correct_bits = sum(1 for orig, ext in zip(original_decoded[:minlen], corrected_bits[:minlen]) if orig == ext)
        bit_accuracy = correct_bits / len(original_decoded) * 100 if len(original_decoded) > 0 else 0.0

    return watermark, bit_accuracy, corrected_bits, original_decoded


#定义不同攻击
def apply_attack(image_path, attack_type, output_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    if attack_type == "rotate":
        #旋转10度
        M = cv2.getRotationMatrix2D((256, 256), 10, 1.0)
        attacked = cv2.warpAffine(img, M, (512, 512), borderValue=(128, 128, 128))
    elif attack_type == "cut":
        #裁剪左上角1/4并放大
        cut = img[0:256, 0:256]
        attacked = cv2.resize(cut, (512, 512))
    elif attack_type == "contrast":
        #对比度增强
        attacked = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
    elif attack_type == "crop":
        #中心裁剪并放大
        crop = img[128:384, 128:384]
        attacked = cv2.resize(crop, (512, 512))
    else:
        attacked = img

    cv2.imwrite(output_path, attacked)


#测试鲁棒性
def test_all_attacks(original_path, watermarked_path, watermark_text, alpha=0.25, repeat=5, coeffs_per_block=6):
    result, accuracy, extracted_bits, original_bits = extract_watermark(
        watermarked_path, original_path, watermark_text, alpha, repeat, coeffs_per_block)

    print("==========================================================")
    print(f"原始提取的水印文本: {result}")
    print(f"比特准确率: {accuracy:.2f}%")

    attacks = {
        "rotate": "旋转10°",
        "cut": "左上裁剪",
        "contrast": "对比度增强",
        "crop": "中心裁剪"
    }

    for atk, desc in attacks.items():
        attacked_img = f"attacked_{atk}.jpg"
        apply_attack(watermarked_path, atk, attacked_img)

        result, accuracy, _, _ = extract_watermark(
            attacked_img, original_path, watermark_text, alpha, repeat, coeffs_per_block)

        print("==========================================================")
        print(f"{atk}后提取的水印文本为: {result}")
        print(f"比特准确率: {accuracy:.2f}%")


if __name__ == "__main__":
    ORIGINAL_IMAGE = "original.jpg"
    WATERMARKED_IMAGE = "watermarked.jpg"
    WATERMARK_TEXT = "SECRET"
    ALPHA = 0.5
    REPEAT = 10
    COEFFS_PER_BLOCK = 6  #每个8x8block嵌入的低频系数个数

    embed_watermark(ORIGINAL_IMAGE, WATERMARK_TEXT, WATERMARKED_IMAGE, ALPHA, REPEAT, COEFFS_PER_BLOCK)
    test_all_attacks(ORIGINAL_IMAGE, WATERMARKED_IMAGE, WATERMARK_TEXT, ALPHA, REPEAT, COEFFS_PER_BLOCK)