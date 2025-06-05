使用说明
cp.py就是最普通的框架结构计算,忽略轴向变形的情况，就直接把ELEMENT类中的include_axial_deformation设置为True即可
cp_skewed_support.py是计算斜支座（包含铰支座和滑移支座）的。同时也可以计算支座自身存在位移的情况，即只需要把constraint里的
（x,y,M）分别给出对应的位移即可
cp_temperature.py是计算包含温度影响的框架结构计算
Node(0, 0.0, 0.0, constraint=[0, 0, 0]) # 第一个0是节点编号，后面两个是坐标，constraint是约束条件，0表示固定，none表示自由，
constraint里的三个值依次对应x,y,M三个方向的约束条件，如果CONSTRAINT里是一个数，就是代表支座本身存在一定的位移
Element(0, 0, 1, E=2e11, A=0.003, I=1.6e-5) 第一个0是单元编号，后面两个是节点编号，即0号单元是从节点0到节点1，
E是弹性模量，A是截面面积，I是惯性矩
K_mod和F_mod是引入所有边界条件后、准备实际求解的方程组
注意在主函数定义时，一定要让ELEMENT和ELEMENT_LOADS长度一致
