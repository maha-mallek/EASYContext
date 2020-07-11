import { Component, OnInit } from '@angular/core';

import { AuthService } from '../../services/auth.service';
import { Router , NavigationEnd } from '@angular/router';
import { JwtHelperService } from '@auth0/angular-jwt';
@Component({
  selector: 'app-header',
  templateUrl: './header.component.html',
  styleUrls: ['./header.component.css']
})
export class HeaderComponent implements OnInit {
  isAuth: boolean;
  mySubscription: any;
  user;

  constructor(public authSerivce : AuthService, private router : Router) {
    this.router.routeReuseStrategy.shouldReuseRoute = function () {
      return false;
    };
    this.mySubscription = this.router.events.subscribe((event) => {
      if (event instanceof NavigationEnd) {
        // Trick the Router into believing it's last link wasn't previously loaded
        this.router.navigated = false;
      }
    });
   }

  ngOnInit(): void {
    console.log(this.authSerivce.isAuthenticated());
    //console.log(this.authSerivce.isLoggedIn());

    this.isAuth = this.authSerivce.isAuthenticated();
    let token= localStorage.getItem('token');
  
  const helper = new JwtHelperService();
  
  const decodedToken = helper.decodeToken(token);
  this.user={
    name:decodedToken.username,
    mail:decodedToken.email,

  }
  }


  ngOnDestroy() {
    if (this.mySubscription) {
      this.mySubscription.unsubscribe();
    }
  }
  logout() {
    //localStorage.removeItem('token');
    this.authSerivce.logout();
    this.isAuth = this.authSerivce.isAuthenticated();
    this.router.navigate(['logout']); 

  }
}
