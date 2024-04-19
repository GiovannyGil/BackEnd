import { NewDiaryEntry, Weather, Visibility } from './types'

/**
 * VERIFICACIONES
 */

// verificacion de strings
const isString = (string: any): boolean => {
  return typeof string === 'string' || string instanceof String
}

// verificacion de fechas
const isDate = (date: string): boolean => {
  return Boolean(Date.parse(date))
}

// verificacion de weather
const isWeather = (param: any): boolean => {
  return Object.values(Weather).includes(param)
}

// verificacion de visibility
const isVisibility = (param: any): boolean => {
  return Object.values(Visibility).includes(param)
}

/**
 * PARSEOS
 */
// parsear comentarios
const parseComment = (commentFromRequest: any): string => {
  if (!isString(commentFromRequest)) { // si no es un string
    throw new Error('Incorrect or missing comment')
  }

  return commentFromRequest
}

// Parsear fechas
const parseDate = (dateFromRequest: any): string => {
  if (!isString(dateFromRequest) || !isDate(dateFromRequest)) { // si no es un string o no es una fecha
    throw new Error('Incorrect or missing date')
  }

  return dateFromRequest
}

// parsear Weather
const parseWeather = (weatherFromRequest: any): Weather => {
  if (!isString(weatherFromRequest) || !isWeather(weatherFromRequest)) { // si no es un string o no es un weather
    throw new Error('Incorrect or missing weather')
  }

  return weatherFromRequest
}

// parsear Visibility
const parseVisibility = (visibilityFromRequest: any): Visibility => {
  if (!isString(visibilityFromRequest) || !isVisibility(visibilityFromRequest)) { // si no es un string o no es un visibility
    throw new Error('Incorrect or missing visibility')
  }

  return visibilityFromRequest
}

const toNewDarieEntry = (object: any): NewDiaryEntry => {
  const newEntry: NewDiaryEntry = {
    date: parseDate(object.date),
    weather: parseWeather(object.weather),
    visibility: parseVisibility(object.visibility),
    comment: parseComment(object.comment)
  }

  return newEntry
}

export default toNewDarieEntry
