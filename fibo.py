numero_max = int(input("Digite a quantidade de elementos na sequencia de Fibonacci: "))

def seq_fibonaci(num_max):
    list_fibo=[0,1]
    cont=1

    while cont <= num_max:
       list_fibo.append(list_fibo[len(list_fibo)-2]+list_fibo[len(list_fibo)-1])
       cont+=1

    return list_fibo

print(seq_fibonaci(numero_max))