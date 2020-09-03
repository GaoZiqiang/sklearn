1   # -*- coding: utf-8 -*-
  2 """
  3 Created on Tue Sep  4 16:58:16 2018
  4 支持向量机代码实现
  5 SMO(Sequential Minimal Optimization)最小序列优化
  6 @author: weixw
  7 """
  8 import numpy as np
  9 #核转换函数（一个特征空间映射到另一个特征空间，低维空间映射到高维空间）
 10 #高维空间解决线性问题，低维空间解决非线性问题
 11 #线性内核 = 原始数据矩阵（100*2）与原始数据第一行矩阵转秩乘积（2*1） =>（100*1）
 12 #非线性内核公式：k(x,y) = exp(-||x - y||**2/2*(e**2))
 13 #1.原始数据每一行与原始数据第一行作差， 
 14 #2.平方   
 15 def kernelTrans(dataMat, rowDataMat, kTup):
 16     m,n=np.shape(dataMat)
 17     #初始化核矩阵 m*1
 18     K = np.mat(np.zeros((m,1)))
 19     if kTup[0] == 'lin': #线性核
 20         K = dataMat*rowDataMat.T
 21     elif kTup[0] == 'rbf':#非线性核
 22         for j in range(m):
 23             #xi - xj
 24             deltaRow = dataMat[j,:] - rowDataMat
 25             K[j] = deltaRow*deltaRow.T
 26         #1*m m*1 => 1*1
 27         K = np.exp(K/(-2*kTup[1]**2))
 28     else: raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
 29     return K
 30         
 31 #定义数据结构体，用于缓存，提高运行速度
 32 class optStruct:
 33     def __init__(self, dataSet, labelSet, C, toler, kTup):
 34         self.dataMat = np.mat(dataSet) #原始数据，转换成m*n矩阵
 35         self.labelMat = np.mat(labelSet).T #标签数据 m*1矩阵
 36         self.C = C #惩罚参数，C越大，容忍噪声度小，需要优化；反之，容忍噪声度高，不需要优化；
 37                    #所有的拉格朗日乘子都被限制在了以C为边长的矩形里
 38         self.toler = toler #容忍度
 39         self.m = np.shape(self.dataMat)[0] #原始数据行长度
 40         self.alphas = np.mat(np.zeros((self.m,1))) # alpha系数，m*1矩阵
 41         self.b = 0 #偏置
 42         self.eCache = np.mat(np.zeros((self.m,2))) # 保存原始数据每行的预测值
 43         self.K = np.mat(np.zeros((self.m,self.m))) # 核转换矩阵 m*m
 44         for i in range(self.m):
 45             self.K[:,i] = kernelTrans(self.dataMat, self.dataMat[i,:], kTup)
 46             
 47 #计算原始数据第k项对应的预测误差  1*m m*1 =>1*1  
 48 #oS：结构数据
 49 #k： 原始数据行索引           
 50 def calEk(oS, k):
 51     #f(x) = w*x + b 
 52     fXk = float(np.multiply(oS.alphas,oS.labelMat).T*oS.K[:,k] + oS.b)
 53     Ek = fXk - float(oS.labelMat[k])
 54     return Ek
 55 
 56 #在alpha有改变都要更新缓存
 57 def updateEk(oS, k):
 58     Ek = calEk(oS, k)
 59     oS.eCache[k] = [1, Ek]
 60     
 61 
 62 #第一次通过selectJrand()随机选取j,之后选取与i对应预测误差最大的j（步长最大）
 63 def selectJ(i, oS, Ei):
 64     #初始化
 65     maxK = -1  #误差最大时对应索引
 66     maxDeltaE = 0 #最大误差
 67     Ej = 0 # j索引对应预测误差
 68     #保存每一行的预测误差值 1相对于初始化为0的更改
 69     oS.eCache[i] = [1,Ei]
 70     #获取数据缓存结构中非0的索引列表(先将矩阵第0列转化为数组)
 71     validEcacheList = np.nonzero(oS.eCache[:,0].A)[0]
 72     #遍历索引列表，寻找最大误差对应索引
 73     if len(validEcacheList) > 1:
 74         for k in validEcacheList:
 75             if k == i:
 76                 continue
 77             Ek = calEk(oS, k)
 78             deltaE = abs(Ei - Ek)
 79             if(deltaE > maxDeltaE):
 80                 maxK = k
 81                 maxDeltaE = deltaE
 82                 Ej = Ek
 83         return maxK, Ej
 84     else:
 85         #随机选取一个不等于i的j
 86         j = selectJrand(i, oS.m)
 87         Ej = calEk(oS, j)
 88     return j,Ej
 89 
 90 #随机选取一个不等于i的索引          
 91 def selectJrand(i, m):
 92     j = i
 93     while (j == i):
 94        j = int(np.random.uniform(0, m))
 95     return j
 96 
 97 #alpha范围剪辑
 98 def clipAlpha(aj, L, H):
 99     if aj > H:
100         aj = H
101     if aj < L:
102         aj = L
103     return aj
104 
105 #从文件获取特征数据，标签数据
106 def loadDataSet(fileName):
107     dataSet = []; labelSet = []
108     fr = open(fileName)
109     for line in fr.readlines():
110         #分割
111         lineArr = line.strip().split('\t')
112         dataSet.append([float(lineArr[0]), float(lineArr[1])])
113         labelSet.append(float(lineArr[2]))
114     return dataSet, labelSet
115 
116 #计算 w 权重系数
117 def calWs(alphas, dataSet, labelSet):
118     dataMat = np.mat(dataSet)
119     #1*100 => 100*1
120     labelMat = np.mat(labelSet).T
121     m, n = np.shape(dataMat)    
122     w = np.zeros((n, 1))    
123     for i in range(m):
124         w += np.multiply(alphas[i]*labelMat[i], dataMat[i,:].T)        
125     return w
126 #计算原始数据每一行alpha,b，保存到数据结构中，有变化及时更新       
127 def innerL(i, oS):
128     #计算预测误差
129     Ei = calEk(oS, i)
130     #选择第一个alpha，违背KKT条件2
131     #正间隔，负间隔
132     if ((oS.labelMat[i] * Ei < -oS.toler) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i] * Ei > oS.toler) and (oS.alphas[i] > 0)):
133         #第一次随机选取不等于i的数据项，其后根据误差最大选取数据项
134         j, Ej = selectJ(i, oS, Ei)
135         #初始化，开辟新的内存
136         alphaIold = oS.alphas[i].copy()
137         alphaJold = oS.alphas[j].copy()
138         #通过 a1y1 + a2y2 = 常量
139         #    0 <= a1,a2 <= C 求出L,H
140         if oS.labelMat[i] != oS.labelMat[j]:
141             L = max(0, oS.alphas[j] - oS.alphas[i])
142             H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
143         else:
144             L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
145             H = min(oS.C, oS.alphas[j] + oS.alphas[i])
146         if L == H : 
147             print ("L == H")
148             return 0
149         #内核分母 K11 + K22 - 2K12
150         eta = oS.K[i, i] + oS.K[j, j] - 2.0*oS.K[i, j]
151         if eta <= 0:
152             print ("eta <= 0")
153             return 0
154         #计算第一个alpha j
155         oS.alphas[j] += oS.labelMat[j]*(Ei - Ej)/eta
156         #修正alpha j的范围
157         oS.alphas[j] = clipAlpha(oS.alphas[j], L, H)
158         #alpha有改变，就需要更新缓存数据
159         updateEk(oS, j)
160         #如果优化后的alpha 与之前的alpha变化很小，则舍弃，并重新选择数据项的alpha
161         if (abs(oS.alphas[j] - alphaJold) < 0.00001):
162             print ("j not moving enough, abandon it.")
163             return 0
164         #计算alpha对的另一个alpha i
165         # ai_new*yi + aj_new*yj = 常量
166         # ai_old*yi + ai_old*yj = 常量 
167         # 作差=> ai = ai_old + yi*yj*(aj_old - aj_new)
168         oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])
169         #alpha有改变，就需要更新缓存数据
170         updateEk(oS, i)
171         #计算b1,b2
172         # y(x) = w*x + b => b = y(x) - w*x
173         # w = aiyixi(i= 1->N求和)
174         #b1_new = y1_new - (a1_new*y1*k11 + a2_new*y2*k21 + ai*yi*ki1(i = 3 ->N求和 常量))
175         #b1_old = y1_old - (a1_old*y1*k11 + a2_old*y2*k21 + ai*yi*ki1(i = 3 ->N求和 常量))
176         #作差=> b1_new = b1_old + (y1_new - y1_old) - y1*k11*(a1_new - a1_old) - y2*k21*(a2_new - a2_old)
177         # => b1_new = b1_old + Ei - yi*(ai_new - ai_old)*kii - yj*(aj_new - aj_old)*kij      
178         #同样可推得 b2_new = b2_old + Ej - yi*(ai_new - ai_old)*kij - yj*(aj_new - aj_old)*kjj
179         bi = oS.b - Ei - oS.labelMat[i]*(oS.alphas[i] - alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j] - alphaJold)*oS.K[i,j]
180         bj = oS.b - Ej - oS.labelMat[i]*(oS.alphas[i] - alphaIold)*oS.K[i,j] - oS.labelMat[j]*(oS.alphas[j] - alphaJold)*oS.K[j,j]
181         #首选alpha i，相对alpha j 更准确
182         if (0 < oS.alphas[i]) and (oS.alphas[i] < oS.C):
183             oS.b = bi
184         elif (0 < oS.alphas[j]) and (oS.alphas[j] < oS.C):
185             oS.b = bj
186         else:
187             oS.b = (bi + bj)/2.0
188         return 1
189     else:
190         return 0
191     
192 #完整SMO核心算法，包含线性核核非线性核，返回alpha,b
193 #dataSet 原始特征数据
194 #labelSet 标签数据
195 #C 凸二次规划参数
196 #toler 容忍度
197 #maxInter 循环次数
198 #kTup 指定核方式
199 #程序逻辑：
200 #第一次全部遍历，遍历后根据alpha对是否有修改判断，
201 #如果alpha对没有修改，外循环终止；如果alpha对有修改，则继续遍历属于支持向量的数据。
202 #直至外循环次数达到maxIter
203 #相比简单SMO算法，运行速度更快，原因是：
204 #1.不是每一次都全量遍历原始数据，第一次遍历原始数据，
205 #如果alpha有优化，就遍历支持向量数据，直至alpha没有优化，然后再转全量遍历，这是如果alpha没有优化，循环结束；
206 #2.外循环不需要达到maxInter次数就终止；
207 def smoP(dataSet, labelSet, C, toler, maxInter, kTup = ('lin', 0)):
208     #初始化结构体类，获取实例
209     oS = optStruct(dataSet, labelSet, C, toler, kTup)
210     iter = 0
211     #全量遍历标志
212     entireSet = True
213     #alpha对是否优化标志
214     alphaPairsChanged = 0
215     #外循环 终止条件：1.达到最大次数 或者 2.alpha对没有优化
216     while (iter < maxInter) and ((alphaPairsChanged > 0) or (entireSet)):
217         alphaPairsChanged = 0
218         #全量遍历 ，遍历每一行数据 alpha对有修改，alphaPairsChanged累加
219         if entireSet:
220             for i in range(oS.m):
221                 alphaPairsChanged += innerL(i, oS)
222                 print ("fullSet, iter: %d i:%d, pairs changed %d" %(iter, i, alphaPairsChanged))
223             iter += 1
224         else:
225             #获取(0，C)范围内数据索引列表，也就是只遍历属于支持向量的数据
226             nonBounds = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
227             for i in nonBounds:
228                 alphaPairsChanged += innerL(i, oS)
229                 print ("non-bound, iter: %d i:%d, pairs changed %d" %(iter, i, alphaPairsChanged))
230             iter += 1
231         #全量遍历->支持向量遍历
232         if entireSet:
233             entireSet = False
234         #支持向量遍历->全量遍历
235         elif alphaPairsChanged == 0:
236             entireSet = True
237         print ("iteation number: %d"% iter)
238         print ("entireSet :%s"% entireSet)
239         print ("alphaPairsChanged :%d"% alphaPairsChanged)
240     return oS.b,oS.alphas
241 
242 #绘制支持向量
243 def drawDataMap(dataArr,labelArr,b,alphas):
244     import matplotlib.pyplot as plt
245     #alphas.A>0 获取大于0的索引列表，只有>0的alpha才对分类起作用
246     svInd=np.nonzero(alphas.A>0)[0]           
247      #分类数据点
248     classified_pts = {'+1':[],'-1':[]}
249     for point,label in zip(dataArr,labelArr):
250         if label == 1.0:
251             classified_pts['+1'].append(point)
252         else:
253             classified_pts['-1'].append(point)
254     fig = plt.figure()
255     ax = fig.add_subplot(111)
256     #绘制数据点
257     for label,pts in classified_pts.items():
258         pts = np.array(pts)
259         ax.scatter(pts[:, 0], pts[:, 1], label = label)
260     #绘制分割线
261     w = calWs(alphas, dataArr, labelArr)
262     #函数形式：max( x ,key=lambda a : b )        #    x可以是任何数值，可以有多个x值
263     #先把x值带入lambda函数转换成b值，然后再将b值进行比较
264     x1, _=max(dataArr, key=lambda x:x[0])
265     x2, _=min(dataArr, key=lambda x:x[0])    
266     a1, a2 = w
267     y1, y2 = (-b - a1*x1)/a2, (-b - a1*x2)/a2
268     #矩阵转化为数组.A
269     ax.plot([x1, x2],[y1.A[0][0], y2.A[0][0]])
270     
271     #绘制支持向量
272     for i in svInd:
273         x, y= dataArr[i]        
274         ax.scatter([x], [y], s=150, c ='none', alpha=0.7, linewidth=1.5, edgecolor = '#AB3319')
275     plt.show()
276     
277      #alpha>0对应的数据才是支持向量，过滤不是支持向量的数据
278     sVs= np.mat(dataArr)[svInd] #get matrix of only support vectors
279     print ("there are %d Support Vectors.\n" % np.shape(sVs)[0])
280     
281 #训练结果    
282 def getTrainingDataResult(dataSet, labelSet, b, alphas, k1=1.3):
283     datMat = np.mat(dataSet)
284     #100*1
285     labelMat = np.mat(labelSet).T
286     #alphas.A>0 获取大于0的索引列表，只有>0的alpha才对分类起作用
287     svInd=np.nonzero(alphas.A>0)[0]
288     sVs=datMat[svInd]
289     labelSV = labelMat[svInd];
290     m,n = np.shape(datMat)
291     errorCount = 0
292     for i in range(m):
293         kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
294         # y(x) = w*x + b => b = y(x) - w*x
295         # w = aiyixi(i= 1->N求和)
296         predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
297         if np.sign(predict)!=np.sign(labelSet[i]): errorCount += 1
298     print ("the training error rate is: %f" % (float(errorCount)/m))
299     
300 def getTestDataResult(dataSet, labelSet, b, alphas, k1=1.3):
301     datMat = np.mat(dataSet)
302     #100*1
303     labelMat = np.mat(labelSet).T
304     #alphas.A>0 获取大于0的索引列表，只有>0的alpha才对分类起作用
305     svInd=np.nonzero(alphas.A>0)[0]
306     sVs=datMat[svInd]
307     labelSV = labelMat[svInd];
308     m,n = np.shape(datMat)
309     errorCount = 0
310     for i in range(m):
311         kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
312         # y(x) = w*x + b => b = y(x) - w*x
313         # w = aiyixi(i= 1->N求和)
314         predict=kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b
315         if np.sign(predict)!=np.sign(labelSet[i]): errorCount += 1    
316     print ("the test error rate is: %f" % (float(errorCount)/m))  
317     
318     

SMO算法实现
