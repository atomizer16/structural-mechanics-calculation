ʹ��˵��
cp.py��������ͨ�Ŀ�ܽṹ����,����������ε��������ֱ�Ӱ�ELEMENT���е�include_axial_deformation����ΪTrue����
cp_skewed_support.py�Ǽ���б֧����������֧���ͻ���֧�����ġ�ͬʱҲ���Լ���֧���������λ�Ƶ��������ֻ��Ҫ��constraint���
��x,y,M���ֱ������Ӧ��λ�Ƽ���
cp_temperature.py�Ǽ�������¶�Ӱ��Ŀ�ܽṹ����
Node(0, 0.0, 0.0, constraint=[0, 0, 0]) # ��һ��0�ǽڵ��ţ��������������꣬constraint��Լ��������0��ʾ�̶���none��ʾ���ɣ�
constraint�������ֵ���ζ�Ӧx,y,M���������Լ�����������CONSTRAINT����һ���������Ǵ���֧���������һ����λ��
Element(0, 0, 1, E=2e11, A=0.003, I=1.6e-5) ��һ��0�ǵ�Ԫ��ţ����������ǽڵ��ţ���0�ŵ�Ԫ�Ǵӽڵ�0���ڵ�1��
E�ǵ���ģ����A�ǽ��������I�ǹ��Ծ�
K_mod��F_mod���������б߽�������׼��ʵ�����ķ�����
ע��������������ʱ��һ��Ҫ��ELEMENT��ELEMENT_LOADS����һ��
��CP_SKEWED_SUPPORT�У�TANGENT������֧��������Լ����Ҳ����б��֧������NORMAL������֧��������Լ����Ҳ����б����֧����
����м�����أ���ֱ����F�����м��Ͼ���
ndof = len(nodes) * 3
F = np.zeros(ndof)
F[4*3+2] = 10.0(�����Y�����������4*3+1)

��֧���ͽṹ�еĽ£�constraint�ǽ�Ȼ�෴��
nodes = [
    Node(0, 0.0, 0.0, constraint=[0, 0, 0]),          # bottom-left, fixed
    Node(1, 0.0, 4.0, constraint=[None, None, None]), # top-left, free
    Node(4, 3.0, 4.0, constraint=[None, None, 0]),    # mid-span hinge, �� fixed
    Node(2, 6.0, 4.0, constraint=[None, None, None]), # top-right, free
    Node(3, 6.0, 0.0, constraint=[0, 0, None]),       # bottom-right, pinned
]

