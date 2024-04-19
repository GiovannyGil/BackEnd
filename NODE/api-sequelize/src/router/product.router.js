import Router from 'express'
import ProductModel from '../model/product.model.js'

const router = Router()

router.get('/products', (req, res) => {
    res.send('PRODUCTS ROUTE')
})
router.get('/products', (req, res) => {
    res.send('PRODUCTS ROUTE')
})
router.post('/products', (req, res) => {
    res.send('PRODUCTS ROUTE')
})
router.put('/products', (req, res) => {
    res.send('PRODUCTS ROUTE')
})
router.delete('/products', (req, res) => {
    res.send('PRODUCTS ROUTE')
})

export default router