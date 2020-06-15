from mulLayer import MulLayer
from addLayer import AddLayer

apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

# layer
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()

# forward
apple_price = mul_apple_layer.forward(apple, apple_num)
orange_price = mul_orange_layer.forward(orange, orange_num)
all_price = add_apple_orange_layer.forward(apple_price, orange_price)
price = mul_tax_layer.forward(all_price, tax)

print("all_price", all_price)
print("price", price)

print("------------------------")
# backward
dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice)
print("dall_price", dall_price)
print("dtax", dtax)
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
print("dapple_price", dapple_price)
print("dorange_price", dorange_price)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print("price", price)
print("dapple_num", dapple_num)
print("dapple", dapple)
print("dorange", dorange)
print("dorange_num", dorange_num)
print("dtax", dtax)


