import { Injectable } from '@angular/core';
import { HttpClient , HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';


@Injectable({
  providedIn: 'root'
})

export class UserService {
  private baseUrl = 'http://localhost:8000/context/download';
  private _b= 'http://localhost:8000/context/downloadkey';
  private _b2= 'http://localhost:8000/api/users/update'
  constructor(private http:HttpClient ) { }

  registerUser(userData): Observable  <any> {
    return this.http.post<any>('http://localhost:8000/api/users/register/',userData);
  }


  posttext(text): Observable  <any> {
    return this.http.post<any>('http://localhost:8000/summarize/text/',text);
  }
  Contextextraction(text):Observable <any>{
    return this.http.post<any>('http://localhost:8000/context/lda/',text)
  }

  downloadrecents(id:number): Observable <any>{
    return this.http.get<any>('http://localhost:8000/context/download/${id}')
  }
  downloadrecent(id: number): Observable<Object> {
    return this.http.get(`${this.baseUrl}/${id}`);
  }

  downloadkeys(id : number): Observable<Object> {
    return this.http.get(`${this._b}/${id}`);
  }
  
  update_informations(id: number, value: any): Observable<Object> {
    return this.http.put(`${this._b2}/${id}`, value);
  }
}
