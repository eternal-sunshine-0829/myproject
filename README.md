## Project 1: 做SM4的软件实现和优化 
#### a): 从基本实现出发 优化SM4的软件执行效率，至少应该覆盖T-table、AESNI以及最新的指令集（GFNI、VPROLD等）（已完成）
- 使用T-table优化sm4思路：将S盒变换与后续的循环移位变换L进行合并，定义四个8bit-->32bit查找表。
- 使用AESNI优化sm4思路：使用SIMD指令进行多个分组数据的并行处理，并使用AESENCLAST实现SM4的S盒变换（借助有限域的同构映射）。
- sm4未优化，使用T-table优化sm4，使用T-table优化sm4三者测试得到时间如下图：
  <img width="1280" height="694" alt="image" src="https://github.com/user-attachments/assets/1fdb0677-64f7-4c34-bbf8-904a4e5b7fa9" />
- 结果分析表格如下：
  
|                | sm4未优化 | T-table优化sm4 | AESNI优化sm4 |
|----------------|----------|---------------|---------------|
| 平均耗时       | 875ns    | 670ns         | 4326ns        |
| 性能           | 基准     | ↑23.4%        | ↓394%         |
| 性能差异原因   | \        | 减少实时计算；提高缓存命中率；消除分支预测 | 转换开销大；测试数据规模较小未能体现优势 |
#### b): 基于SM4的实现，做SM4-GCM工作模式的软件优化实现（未完成）
