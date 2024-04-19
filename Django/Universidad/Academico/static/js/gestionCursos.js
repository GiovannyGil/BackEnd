// la funcion se cargara con la pagina, "inmediatamente de cargue la pagina"
(function(){
    const btnEliminacion=document.querySelectorAll(".btnEliminacion");

    btnEliminacion.forEach(btn=>{
        btn.addEventListener('click', (e)=>{
            const confirmacion = confirm('¿Seguro de eliminar el curso?');
            if(!confirmacion){
                e.preventDefault();
            }
        })
    })
})();