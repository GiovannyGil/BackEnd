import { diaryEntry, NonSensitiveInfoDiaryEntry, NewDiaryEntry } from '../types'
import diaryData from './diaries.json'

const diaries: diaryEntry[] = diaryData as diaryEntry[] // castear a tipo diaryEntry con asercion de tipoes

export const getEntries = (): diaryEntry[] => diaries // obtener todos los diarios

// busqueda por id
export const findById = (id: number): NonSensitiveInfoDiaryEntry | undefined => {
  const entry = diaries.find(d => d.id === id)
  if (entry != null) {
    const { comment, ...restOfDiary } = entry // desestructurar el diario y obtener el campo comment
    return restOfDiary // retornar el diario sin el campo comment
  }

  return undefined
}

export const getEntriesWithoutSensitiveInfo = (): NonSensitiveInfoDiaryEntry[] => {
  return diaries.map(({ id, date, weather, visibility }) => ({
    id,
    date,
    weather,
    visibility
  }))
} // obtener todos los diarios sin el campo comment

// agregar un nuevo diario
export const addDiary = (newDiaryEntry: NewDiaryEntry): diaryEntry => {
  const newDiary = {
    id: Math.max(...diaries.map(d => d.id)) + 1, // obtener el id mas grande y sumarle 1
    ...newDiaryEntry // agregar los datos del nuevo diario
  }

  diaries.push(newDiary) // agregar el nuevo diario al arreglo de diarios

  return newDiary
} // agregar un nuevo diario
