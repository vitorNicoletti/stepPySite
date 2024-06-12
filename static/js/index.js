/*Mostrar menu*/
var btn = document.querySelector('#btn');
var navbar =  document.querySelector('.navbar');

var menu_aberto = false;

function showMenu(){

    if(!menu_aberto){
        navbar.style.width = '250px';
        navbar.style.transition = '1s';
        menu_aberto = true;
    }
    else {
        navbar.style.width = '0px';
        navbar.style.transition = '1s';
        menu_aberto = false;
    }
}

$(document).ready(function() {
    $('#solve-limit').click(function(event) {
        event.preventDefault();

        var user_input = $('#input').val();

        $.ajax({
            type: 'POST',
            url: '/',
            data: {input: user_input},
            success: function(response) {
                $('#output-solution').empty();
                if (response.results) {
                    $.each(response.results, function(index, result) {
                        var resultDiv = $('<div class="result"></div>');
                        $.each(result, function(_, line) {
                            resultDiv.append('<div class="line">$$' + line + '$$</div>');
                        });
                        $('#output-solution').append(resultDiv);
                    });
                }
                if (response.images) {
                    $.each(response.images, function(_, image) {
                        if (image) {
                            $('#output-solution').append('<div class="img-solution"><img src="' + image + '" alt="image"></div>');
                        }
                    });
                }
                if (response.error) {
                    $('#output-solution').empty();
                    $('#output-solution').append('<div class="error">' + response.error + '</div>');
                }

                MathJax.typeset();

                $('#second-page').get(0).scrollIntoView({ behavior: 'smooth' });
            },
            error: function(xhr, status, error) {
                console.log(xhr.responseText);
            }
        });

        return false;
    });
});











