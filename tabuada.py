tabuada = int(input("Digite qual tabuada deseja calcular: "))

tab_list=[1,2,3,4,5,6,7,8,9,10]

for elemento in tab_list:
    print ("{0} X {1} = {2}".format(tabuada, elemento, elemento*tabuada))