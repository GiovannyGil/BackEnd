function sum(a, b){
    return a + b
}


// exportar de forma clasica con NODEjs
modulo.export = sum 
// o así
modulo.export = {
    sum
}