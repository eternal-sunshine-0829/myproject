# Project 1: 做SM4的软件实现和优化 
## a): 从基本实现出发 优化SM4的软件执行效率，至少应该覆盖T-table、AESNI以及最新的指令集（GFNI、VPROLD等）（已完成）
- 使用T-table优化sm4思路：将S盒变换与后续的循环移位变换L进行合并，定义四个8bit-->32bit查找表。
- 使用AESNI优化sm4思路：使用SIMD指令进行多个分组数据的并行处理，并使用AESENCLAST实现SM4的S盒变换（借助有限域的同构映射）
- sm4未优化，使用T-table优化sm4，使用T-table优化sm4三者测试得到时间如下图：
  <img width="1280" height="694" alt="image" src="https://github.com/user-attachments/assets/1fdb0677-64f7-4c34-bbf8-904a4e5b7fa9" />

