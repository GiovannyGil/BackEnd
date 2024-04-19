import Express from 'express'
import * as diaryServices from '../services/diaryServices'
import { NewDiaryEntry } from '../types'
import toNewDarieEntry from '../utils'

const router = Express.Router()

// GET /api/diaries
router.get('/', (_req, res) => {
  res.send(diaryServices.getEntriesWithoutSensitiveInfo()) // enviar todos los diarios sin el campo comment
})

// GET /api/diaries/:id
router.get('/:id', (req, res) => { // :id es un parametro
  const diary = diaryServices.findById(Number(req.params.id)) // obtener el diario por id

  return (diary != null) // si existe el diario
    ? res.send(diary) // si existe el diario, enviarlo
    : res.sendStatus(404) // si no existe, enviar un 404
})

router.post('/', (req, res) => { // POST /api/diaries
  try {
    const newDiaryEntry: NewDiaryEntry = toNewDarieEntry(req.body) // parsear el nuevo diario

    const addedDiaryEntry = diaryServices.addDiary(newDiaryEntry) // agregar el nuevo diario

    res.json(addedDiaryEntry) // enviar el nuevo diario
  } catch (e) { // si hay un error
    if (e instanceof Error) { // si el error es de tipo Error
      res.status(400).send(e.message) // enviar un 400 y el mensaje de error
    } else {
      // handle other types of 'e' here
    }
  }
})

export default router // exportar el router para usarlo en index.ts
