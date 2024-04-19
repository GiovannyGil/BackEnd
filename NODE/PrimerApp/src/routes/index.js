import { Router } from "express"

const router = Router()

router.get('/', (req, res) => res.render('index.ejs', {title: 'First Web With NODE'})) // cuando visite la página principal devuleva esta funcion
router.get('/about', (req, res) => res.render('about.ejs', {title: 'About Me'})) // ruta a about
router.get('/contact', (req, res) => res.render('contact.ejs', {title: 'Contact Page'})) // ruta a contact


export default router