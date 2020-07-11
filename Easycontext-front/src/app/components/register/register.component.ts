import { Component, OnInit , } from '@angular/core';
import { FormGroup, FormControl, FormBuilder, Validators } from '@angular/forms';
import { UserService } from 'src/app/services/user.service';
import { Router } from '@angular/router';

import { ToastrService } from 'ngx-toastr';
@Component({
  selector: 'app-register',
  templateUrl: './register.component.html',
  styleUrls: ['./register.component.css'],
  providers : [UserService]
})
export class RegisterComponent implements OnInit {

  registerForm: FormGroup;
  register;

  constructor(private fb: FormBuilder , private userService : UserService, private router : Router,private toastr: ToastrService) {
    let formControls = {
      username: new FormControl('', [
        Validators.required,
        Validators.pattern('[a-zA-Z][a-zA-Z][^0-9#&<>\"~;$^%{}?]{1,20}$')
      ]),

      email: new FormControl('', [
        Validators.required,
        Validators.email
      ]),
      password: new FormControl('', [
        Validators.required,
        Validators.minLength(8)
      ]),
      repassword: new FormControl('', [
        Validators.required,
      ]),

    }

    this.registerForm = fb.group(formControls);
  }

  get username() { return this.registerForm.get('username'); }

  get email() { return this.registerForm.get('email'); }
  get password() { return this.registerForm.get('password'); }
  get repassword() { return this.registerForm.get('repassword'); }


  ngOnInit() {
    this.register = {
      username: '',
      password: '',
      email: ''


    };
  }

  /*Register() {
    let userData= {'username': this.registerForm.value.username,
    'password' : this.registerForm.value.password,

  }
  console.log(userData);
    //console.log(this.registerForm.value);
    let ud1 = JSON.stringify(userData);
    console.log(ud1);

    this.userService.registerUser(ud1).subscribe(
      response =>{
        alert('User'+ this.registerForm.value('firstname') + 'has been created successfully!')
      },
      error => console.log('error',error)
    );

  }*/
  
  Register(){
    this.userService.registerUser(this.register)
    .subscribe(
      response => {

        
        console.log(this.register);
        this.toastr.success('You Have Been Registred Successfully !');/*success/error/warning/info/show()*/
        this.router.navigateByUrl('login');

      },
      error => {
        console.log('error',error);
        this.toastr.warning('There is a user with the same Username, please try to change yours !');
    }
      
    );
  }
}