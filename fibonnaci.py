#fibonnaci
#1,2,3,5,8...

num_max = int(input("Max number for fibonnaci sequence: "))

def fibo_sequence(num_max = 100):
    fib_list=[1,2]
    size = len(fib_list)

    while fib_list[size-1] < num_max:
       fib_list.insert(size, fib_list[size-1]+fib_list[size-2])
       size = len(fib_list)
    
    return fib_list

print("the sequency is "+str(fibo_sequence(num_max))+" .")