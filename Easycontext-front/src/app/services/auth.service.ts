import { Injectable } from '@angular/core';
import { HttpClientJsonpModule, HttpClient } from '@angular/common/http';
import {JwtHelperService} from '@auth0/angular-jwt';
@Injectable({
  providedIn: 'root'
})
export class AuthService {
  public jwtHelper: JwtHelperService = new JwtHelperService();
  isAuth : boolean;
  constructor(private http: HttpClient) { 
  }
  ngOnInit(): void {
        
    //Called after the constructor, initializing input properties, and the first call to ngOnChanges.
    //Add 'implements OnInit' to the class.
  }
  loginUser(user){
    return this.http.post('http://localhost:8000/api/auth/login/', user);
  }
  isLoggedIn() : boolean {
    let token = localStorage.getItem('token');
    if (token) {
      return true;
    } else {
      return false;
    }
  }
  public isAuthenticated(): boolean {
    const token = localStorage.getItem('token');
    // Check whether the token is expired and return
    // true or false
    //this.isAuth = !this.jwtHelper.isTokenExpired(token);
    return !this.jwtHelper.isTokenExpired(token);
  }


  
  /*isAdmin() {
    let token = localStorage.getItem('token');
    const helper = new JwtHelperService();
    const decodedToken = helper.decodeToken(token);

    if (token) {
      let role = decodedToken.role;

      if (role == "admin") {
        return true;
      } else {
        return false;
      }
    } else {
      return false;
    }

  }

  isStudent() {
    let token = localStorage.getItem('token');
    const helper = new JwtHelperService();
    const decodedToken = helper.decodeToken(token);

    if (token) {
      let role = decodedToken.role;

      if (role == "student") {
        return true;
      } else {
        return false;
      }
    } else {
      return false;
    }
  }*/

  logout() {
    localStorage.clear();
  }

}
