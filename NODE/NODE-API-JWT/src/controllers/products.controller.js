// importamos el modelo de productos
import Product from "../models/Product.js";

// funciones crud para productos

export const createProduct = async (req, res) => { 
    console.log(req.body)

    const { name, category, price, imgURL } = req.body

    const newProduct = new Product({name, category, price, imgURL})

    const ProductSave = await newProduct.save()
    res.status(201).json(ProductSave)


}

export const GetProducts = async (req, res) => {
    try{
        const products = await Product.find()
        res.json(products)
    } catch (error){
        console.log('no se pudo traer los productos')
        console.log(error)
        res.json(error)
    }
}

export const GetProductById = async (req, res) => {
    try{
        const product = await Product.findById(req.params.productId)
        res.status(200).json(product)
    } catch (error) {
        console.log('algo salio mal trayendo el producto por ID', error)
        res.status(404).json('Product not found')
    }
}

export const UpdateProductsById = async (req, res) => {
    try{
        const updatedProduct = await Product.findByIdAndUpdate(req.params.productId, req.body, { new: true })
        res.status(200).json(updatedProduct)
    } catch (error) {
        console.log('algo salio mal actualizando el producto por ID', error)
        res.status(404).json('Product not found')
    }
}

export const DeleteProductById = async (req, res) => {
    try {
        const deletedProduct = await Product.findByIdAndDelete(req.params.productId)
        res.status(204).json()
    } catch (error) {
        console.log('algo salio mal eliminando el producto por ID', error)
        res.status(404).json('Product not found')
    }
}
