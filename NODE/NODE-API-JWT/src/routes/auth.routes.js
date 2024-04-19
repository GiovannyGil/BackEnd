import { Router } from "express"

// importar el controlador de autenticacion
import * as authCtrl from "../controllers/auth.controller"
import { verifySignup } from "../middlewares"


const router = Router()

// rutas para logeo y autenticacion

router.post('/signup', [verifySignup.checkDuplicateUsernameOrEmail, verifySignup.checkRolesExisted], authCtrl.signUp)
router.post('/signin', authCtrl.signIn)


export default router