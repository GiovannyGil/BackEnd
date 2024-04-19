// funcion para el nav -> responsive
function updatemenu() {
    if (document.getElementById('responsive-menu').checked == true) {
      document.getElementById('menu').style.borderBottomRightRadius = '0';
      document.getElementById('menu').style.borderBottomLeftRadius = '0';
    }else{
      document.getElementById('menu').style.borderRadius = '10px';
    }
  }
  

// modal
// $('.openmodale').click(function (e) {
//     e.preventDefault();
//     $('.modale').addClass('opened');
// });
// $('.closemodale').click(function (e) {
//     e.preventDefault();
//     $('.modale').removeClass('opened');
// });

function agregarCliente() {
    htmlModal = document.getElementById('modal')
    htmlModal.setAttribute('class', 'modale opened')
}

function CerrarModal() {
    htmlModal = document.getElementById('modal')
    htmlModal.setAttribute('class', 'modale')
}