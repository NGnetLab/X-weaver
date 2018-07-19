#!/usr/bin/python
# -*- coding: utf-8 -*
# 目前根据端口先默认fatTree,拓扑结果为一个环
import copy
import random
import numpy as np
import time
from util import *
random.seed(2)

BANDWIDTH = 1250
SPEED = 15
RTT = 0.1
SSRENTSH = 512
# maxm = 8
MICEFLOW = 1000
# result = []  # 最短路径输出结果
# stack = []  # 最短路径辅助栈
# TcpWindow = []
# demandTime = np.zeros([maxm,maxm])
# demandcopy = np.zeros([maxm,maxm])
class pystream :
    def __init__(
            self,
            timeLength,
            topo = None,
            demand = None,
            TcpFlag=0
    ):
        self.timeLength = timeLength
        self.topo       = topo
        self.demand     = demand
        self.TcpWindow  = []
        self.TcpFlag    = TcpFlag
        self.demandNext = demand
        self.result     = []
        self.stack      = []

        self.littleFlag = np.zeros([len(self.demand),len(self.demand)])
        self.bigFlag = np.zeros([len(self.demand),len(self.demand)])
        self.micecount = 0
        self.FatTreeFlag = False
    def Reset (self ,topo, demand) :
        self.topo = topo
        self.demand = demand
        self.TcpFlag = 0
        self.timeLength = 10000

        self.littleFlag, self.bigFlag = self.demandHandle(self.demand)
        self.streamMain()
    def InputUpdate (self,topo,change_interval) :
        self.topo = topo
        self.demand = self.demandNext
        self.TcpFlag= 1
        self.timeLength = change_interval
        # self.littleEmpty = False
        # pystream.streamMain(self)
    def streamMain(self):
        # topoNormal = self.normalTopo(self.topo)
        topoNormal = self.topo
        # global demandcopy
        # demandcopy = copy.deepcopy(demand)
        self.result = []  # 最短路径输出结果
        self.stack = []  # 最短路径辅助栈

        # 流
        Demand = self.demand
        for iii in range(len(Demand)) :
            for jjj in range(len(Demand)) :
                if Demand[iii][jjj] < 50 :
                    Demand[iii][jjj] = 0

        # 底层拓扑
        topofix = fatTreeInit(4)
        # print(topoDeBrejin)
        # 初始化任务
        # taskLitte = self.demandToTask(littleDemand)
        taskBig = self.demandToTask(Demand)
        # 算各自最短路
        topoNormalcopy = copy.deepcopy(topoNormal)
        topoDeBrejincopy = copy.deepcopy(topofix)
        # shortRouterLittle = self.taskToFlow1(taskLitte, topoDeBrejincopy)
        # a = time.time()
        shortRouterBig = self.taskToFlow1(taskBig, topoNormalcopy)
        # b = time.time()
        # print(b - a)
        # exit()
        # print("ttt", topoDeBrejin)
        # flowInfo = self.getFlowId(shortRouterLittle)
        # littleflowCount = len(flowInfo)
        # print(littleflowCount)
        # self.micecount = littleflowCount
        # print("qqqqqq",littleflowCount)

        # flowInfo = self.getFlowId(shortRouterLittle)
        # flowNumInfo = self.getFlowNum(shortRouterLittle, taskLitte)
        flowInfo = self.getFlowId(shortRouterBig)
        flowNumInfo = self.getFlowNum(shortRouterBig, taskBig)
        # flowInfo.extend(flowBigInfo)
        # flowNumInfo.extend(flowBigNum)

        # print("aaaaaaa", sum(flowNumInfo))
        # print("-------", len(flowNumInfo))
        # var_dump(len(flowNumInfo))
        edgeHashInfo = self.edgeHash(topoNormal)
        # var_dump(edgeHashInfo)
        edgeFlowInfo = self.flowToEdge(flowInfo, edgeHashInfo, flowNumInfo)
        # global TcpWindow
        self.TcpWindow = self.intitialTcpWindow(flowNumInfo, self.TcpFlag, self.TcpWindow)
        # print(flowInfo)
        # print(flowNumInfo)
        spendTime, demandRest,dele = self.step_time(self.timeLength, edgeFlowInfo, flowInfo, flowNumInfo, edgeHashInfo, topoNormal,
                                          self.TcpWindow, self.micecount,topofix, True)  # 获得时间片后的情况\
        pp = time.time()
        self.deleTcpWindow(dele,self.TcpWindow)
        # np.savetxt('new.csv', demandTime, delimiter=',')
        # demandNext = self.OutPutDemand(flowInfo, demandRest)
        # resultDemand = np.array(demandNext)
        # demandNP = np.array(self.demand)
        # print(resultDemand)
        # print(TcpWindow)
        # self.demandNext = resultDemand
        # return spendTime, resultDemand, demandNP
        return spendTime, self.demand, self.topo
    def deleTcpWindow(self,dele,TcpWindow):
        # print("sss",dele)
        if len(dele) == 0:
            return
        dele = np.sort(dele)
        for i in range(len(dele)):
            del TcpWindow[dele[i]-i]
    def normalTopo(self,topo) :
        topo_b = copy.deepcopy(topo)
        for i in range(len(self.demand)) :
            for j in range (len(self.demand)) :
                if topo_b[i][j] > 0 :
                    topo_b[i][j] = 1
        return topo_b
    # Dijkstra求最短路，求出一个包含最短的前驱节点的数组
    def Dijkstra(self,topoNormal, v=0):
        maxInt = 1000000000
        dist = []  # 记录最短距离
        pre = []  # 纪录上一节点
        queue = []  # 队列
        # 初始化不可到达点
        for i in range(len(topoNormal)):
            for j in range(len(topoNormal)):
                if j != i and topoNormal[i][j] == 0:
                    topoNormal[i][j] = maxInt
        for i in range(len(topoNormal)):
            templist = []
            dist.append(topoNormal[v][i])
            queue.append(False)
            if dist[i] == maxInt:
                templist.append(-1)
                pre.append(templist)
            else:
                templist.append(v)
                pre.append(templist)
        dist[v] = 0
        queue[v] = True
        for i in range(1, len(topoNormal)):
            mindist = maxInt  # 当前最小值
            u = v  # 前驱节点
            for j in range(len(topoNormal)):
                if queue[j] == False and dist[j] < mindist:
                    u = j
                    mindist = dist[j]
            queue[u] = True
            for j in range(len(topoNormal)):
                if queue[j] == False and topoNormal[u][j] < maxInt:
                    if dist[u] + topoNormal[u][j] < dist[j]:  # 在通过新加入的u点路径找到离v0点更短的路径
                        dist[j] = dist[u] + topoNormal[u][j]  # 更新dist
                        pre[j][0] = u  # 纪录前驱顶点
                    if dist[u] + topoNormal[u][j] == dist[j]:  # 在通过新加入的u点路径找到离v0点更短的路径
                        if pre[j][0] == -1:
                            pre[j][0] = u
                        elif u not in pre[j]:
                            pre[j].append(u)
        return pre


    # 解析前驱数组，求出最短路径
    def dfs(self,pre, targetNode, sourceNode):
        # return 一个数组，每一列纪录一条最短路径
        for i in range(len(pre[targetNode])):
            self.stack.append(pre[targetNode][i])
            if pre[targetNode][i] == sourceNode:
                self.result.append(copy.deepcopy(self.stack))
                self.stack.pop()
                return
            else:
                self.dfs(pre, pre[targetNode][i], sourceNode)
            self.stack.pop()


    # 将每条边分配一个id
    def edgeHash(self,topoNormal):
        outPut = []
        for i in range(len(topoNormal)):
            for j in range(len(topoNormal)):
                if topoNormal[i][j] == 1:
                    temp = str(i) + "," + str(j)
                    outPut.append(temp)
        return outPut


    # 将任务划分成流，其中选用了ECMP的规则
    def taskToFlow1(self,task, topoNormal):
        shortRouter = []
        last_first_node = 10000000000
        for i in range(len(task)):
            self.result = []
            fromTo = task[i][0].split(",")
            if(int(fromTo[0]) != last_first_node):
                preArray = self.Dijkstra(topoNormal, int(fromTo[0]))
            self.dfs(preArray, int(fromTo[1]), int(fromTo[0]))
            # shortRouter.append(self.result)
            out =  []
            if len(self.result) > (len(self.demand)/20):
                for jj in range(int(len(self.demand)/20)):
                    choice = random.choice(self.result)
                    out.append(choice)
                shortRouter.append(out)
            else:
                shortRouter.append(self.result)
            for t in range(len(shortRouter[i])):
                shortRouter[i][t].insert(0, int(fromTo[1]))
            last_first_node = int(fromTo[0])
        return shortRouter

    # 分配流ID
    def getFlowId(self,shortRouter):
        flowInfo = []
        for num in range(len(shortRouter)):
            for i in range(len(shortRouter[num])):
                flowInfo.append(shortRouter[num][i])
        return flowInfo


    # 分配每条流的大小
    def getFlowNum(self,shortRouter, task):
        flowNumInfo = []
        for num in range(len(shortRouter)):
            for i in range(len(shortRouter[num])):
                flowNumInfo.append(task[num][1] / len(shortRouter[num]))
        return flowNumInfo


    # 将流分配到每条边上
    def flowToEdge(self,flowInfo, edgeHashInfo, flowNumInfo):
        edgeFlowInfo = {}
        for key in range(len(flowInfo)):
            for i in range(len(flowInfo[key]) - 1):
                # if flowInfo[key][i] > flowInfo[key][i + 1]:
                tempData = str(flowInfo[key][i]) + "," + str(flowInfo[key][i + 1])
                # else:
                #     tempData = str(flowInfo[key][i + 1]) + "," + str(flowInfo[key][i])
                if tempData in edgeHashInfo:
                    k = edgeHashInfo.index(tempData)
                    if k in edgeFlowInfo:
                        edgeFlowInfo[k] = edgeFlowInfo[k] + "," + str(key)
                    else:
                        edgeFlowInfo[k] = str(key)
        return edgeFlowInfo

    def intitialTcpWindow(self,flowNumInfo,TcpFlag,TcpWindow) :
        # flowNumInfo的key为每条流的id
        # 正式系统中每次初始化后启动窗口值为随机值，现在手动规定的，方便观察。
        if TcpFlag == 0 or len(TcpWindow) == 0:
            TcpWindow = []
            for i in range(len(flowNumInfo)):
                TcpWindow.append(1)
        else :
            for j in range(len(flowNumInfo)):
                if j >= len(TcpWindow) :
                    TcpWindow.append(TcpWindow[ j % len(TcpWindow)])
        return TcpWindow
    def step_time(self,change_interval ,edgeFlowInfo, flowInfo,flowNumInfo,edgeHashInfo, topo,TcpWindow, topolitteCount,topoDeBrejin,FatTreeFlag):
        dele = []
        # print("zuixiaoliu ",topolitteCount)
        # print("ssssss", topoDeBrejin)
        T = change_interval
        # 下面开始主逻辑，循环
        # print("zuiduanlu",len(flowNumInfo))
        for curtime in range(T):
            # 先将其增大，在判断是否拥塞，决定是否减半
            for i in range(len(TcpWindow)):
                if TcpWindow[i] < SSRENTSH:  # 暂时将sstrensh 值定位8
                    if TcpWindow[i] == 0 :
                        continue;
                    elif TcpWindow[i] < 1 and TcpWindow[i] > 0 :
                        TcpWindow[i] = 2
                    else :
                        TcpWindow[i] *= 2
                else:
                    TcpWindow[i] += 1
            tranSum = 0
            topoBandWid = 0
            for num,numVal in edgeFlowInfo.items():
                tranSum = 100000000000  # 保证进循环
                salt = 0
                salt1 = 0
                topoBandWidthArr = edgeHashInfo[num].split(",")
                topoBandWid = topo[int(topoBandWidthArr[0])][int(topoBandWidthArr[1])]
                # print(topoDeBrejin)
                # exit()
                if not FatTreeFlag :
                    topoDeBrejinWid = topoDeBrejin[int(topoBandWidthArr[0])][int(topoBandWidthArr[1])]
                while tranSum > BANDWIDTH :  # 暂时每条边的最大传输值为 20
                    flowArray = edgeFlowInfo[num].split(",")  # 将这条边的flowid上的所有流分割出来
                    BigFlowArray = []
                    LittFlowArray = []
                    for y in range(len(flowArray)) :
                        if int(flowArray[y]) >= topolitteCount :
                            BigFlowArray.append(flowArray[y])
                        else :
                            LittFlowArray.append(flowArray[y])
                    # print(num)
                    if not FatTreeFlag :
                        tranLittleSum = 0
                        for j in range (len(LittFlowArray)) :
                            tranLittleSum += int(TcpWindow[int(LittFlowArray[j])] * SPEED)
                            if tranLittleSum > BANDWIDTH * topoDeBrejinWid and len(LittFlowArray) != 0:
                                salt1 += 1
                                decreaseFlow = ((curtime + num + j + 1 + salt1) * 24036583) % (len(LittFlowArray))
                                TcpWindow[int(LittFlowArray[decreaseFlow])] /= 2
                                # print("xiaoliu",TcpWindow[int(LittFlowArray[decreaseFlow])])
                            else :
                                break
                    tranSum = 0
                    for j in range(len(BigFlowArray)):
                        tranSum += int(TcpWindow[int(BigFlowArray[j])] * SPEED)  # 计算目前情况下这条边上所有流大小
                    if tranSum <= BANDWIDTH * topoBandWid:
                        break
                    elif len(BigFlowArray) != 0:
                        salt += 1
                        decreaseFlow = ((curtime+num+j+1+salt) *  24036583) % (len(BigFlowArray))
                        TcpWindow[int(BigFlowArray[decreaseFlow])] /= 2
                        # print("daliu",TcpWindow[int(BigFlowArray[decreaseFlow])])


                # topo[int(topoBandWidthArr[0])][int(topoBandWidthArr[1])] -=  tranSum/BANDWIDTH
            # print(tranSum,"-----", topoBandWid*BANDWIDTH, "        ",curtime)
            # print("--------------------------------------start-------------------------------------------")
            # print(TcpWindow)
            # print("---------------------------------------end--------------------------------------------")
            #此时已经计算完了 无拥塞下的各个TCP的窗口

            b = time.time()

            for idx in range(len(flowNumInfo)):
                if (flowNumInfo[idx] == 0):
                    continue
                if flowNumInfo[idx] > (TcpWindow[idx] * SPEED * RTT):
                    flowNumInfo[idx] -= (TcpWindow[idx] * SPEED * RTT)
                else:  # 该条流传完了
                    # todo : demand完成时间输出
                    # demandcopy[int(flowInfo[idx][0])][int(flowInfo[idx][-1])] -= float(flowNumInfo[idx])
                    # if demandcopy[int(flowInfo[idx][0])][int(flowInfo[idx][-1])] <= 0 and demandTime[int(flowInfo[idx][0])][
                    #     int(flowInfo[idx][-1])] == 0:
                    #     demandTime[int(flowInfo[idx][0])][int(flowInfo[idx][-1])] = curtime

                    flowNumInfo[idx] = 0
                    keys = []
                    for idkey, v in edgeFlowInfo.items():
                        keys.append(idkey)
                    # print("*****************************************************************")
                    # print(edgeFlowInfo)
                    # if curtime == 99 :
                    #     print(keys)
                    # print("******************************************************************")
                    for key in range(len(keys)):
                        clearArray = edgeFlowInfo[keys[key]].split(",")
                        idxStr = str(idx)
                        if idxStr in clearArray:
                            needDelete = clearArray.index(idxStr)
                            # print(clearArray[needDelete], "你被传完了，时间是", curtime, "剔除的边是", key)
                            del clearArray[needDelete]
                            if len(clearArray) == 0:
                                del edgeFlowInfo[keys[key]]
                            else:
                                clearArrayStr = ",".join(clearArray)
                                edgeFlowInfo[keys[key]] = clearArrayStr
                    if idx not in dele:
                        dele.append(idx)
                # print(flowNumInfo)
                flowSum = 0
                for i in range(len(flowNumInfo)):
                    if flowSum > 0:
                        break
                    flowSum += flowNumInfo[i]

                if flowSum == 0 :
                    # print(flowNumInfo)
                    return curtime+1,flowNumInfo,dele#,demandTime
        # print(flowNumInfo)
        return change_interval, flowNumInfo,dele #,demandTime
    def OutPutDemand (self,flowInfo,demandRest):
        #组装demand begin
        demandNext = np.zeros([len(self.demand),len(self.demand)])
        for k in range(len(demandRest)) :
            if demandRest[k] != 0 :
                # if flowInfo[k][0] > flowInfo[k][-1] :
                #     demandNext[flowInfo[k][-1]][flowInfo[k][0]] += demandRest[k]
                # else :
                try :
                    demandNext[flowInfo[k][-1]][flowInfo[k][0]] += demandRest[k]
                except IndexError:
                    # print(flowInfo[k][0],flowInfo[k][-1],demandRest[k])
                    print("bug-")
        return demandNext
        #组装demand end
    def demandToTask (self,demand) :
        task = []
        for i in range (len(self.demand)) :
            for j in range (len(self.demand)) :
                if demand[i][j] != 0 :
                    temp = [str(i)+","+str(j),demand[i][j]]
                    task.append(temp)
        return task
    if __name__ == '__main__':
        file_path = "../../s_demand0.txt"
    def demandHandle(self,demand) :
        littleFlag = np.zeros([len(self.demand),len(self.demand)])
        bigFlag = np.zeros([len(self.demand),len(self.demand)])
        for i in range(len(self.demand)) :
            for j in range(len(self.demand)) :
                if demand[i][j] < MICEFLOW :  # 小流阈值
                    littleFlag[i][j] = 1
                    bigFlag[i][j] = 0
                else :
                    littleFlag[i][j] = 0
                    bigFlag[i][j] = 1
        return littleFlag,bigFlag
    def getDemand(self,demand,flag):
        res = np.zeros([len(self.demand),len(self.demand)])
        for i in range(len(self.demand)):
            for j in range (len(self.demand)) :
                if demand[i][j] != 0 and flag[i][j] == 1 :
                    res[i][j] = demand[i][j]
        return res

    def FatTree(self, n, hh, demand):
        topo = util.fatTreeInit(n)
        demandcopy = np.zeros([hh, hh])
        for i in range(hh):
            for j in range(hh):
                if i < len(self.demand) and j < len(self.demand):
                    demandcopy[i][j] = float(demand[i][j])
        self.TcpWindow = []
        self.timeLength = 10000000
        self.demand = demandcopy
        self.topo = topo
        self.TcpFlag = 0
        self.littleFlag, self.bigFlag = self.demandHandle(self.demand)
        self.FatTreeFlag = True
        spendTime, demandNext, demandCur = self.streamMain()
        print("fatTree SpendTime: \t",spendTime)