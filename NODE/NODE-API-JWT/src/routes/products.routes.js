import { Router } from "express"
// importar el controlador de productos
import * as productsController from '../controllers/products.controller.js'

// importar middleware de autenticacion -> verificar el token
import { authJwt } from '../middlewares'

const router = Router()

// definir las rutas para productos con sus respectivos controladores
router.get('/', productsController.GetProducts)
router.get('/:productId', productsController.GetProductById)
router.post('/', [authJwt.verifyToken, authJwt.isAdmin, authJwt.isModerator], productsController.createProduct)
router.put('/:productId', [authJwt.verifyToken, authJwt.isAdmin, authJwt.isModerator], productsController.UpdateProductsById)
router.delete('/:productId', [authJwt.verifyToken, authJwt.isAdmin, authJwt.isModerator], productsController.DeleteProductById)


export default router