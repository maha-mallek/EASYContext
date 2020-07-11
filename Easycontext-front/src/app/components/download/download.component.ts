import { Component, OnInit , ViewChild, ElementRef} from '@angular/core';
import { JwtHelperService } from '@auth0/angular-jwt';
import { UserService } from 'src/app/services/user.service';
import { ResolvedStaticSymbol } from '@angular/compiler';
import { DomSanitizer } from '@angular/platform-browser';

import * as jsPDF from 'jspdf'
import 'jspdf-autotable'
import { ToastrService } from 'ngx-toastr';
@Component({
  selector: 'app-download',
  templateUrl: './download.component.html',
  styleUrls: ['./download.component.css']
})
export class DownloadComponent implements OnInit {
  @ViewChild('htmlData') htmlData:ElementRef;
  id:number;
  resultat;
  keyss=[];
  res;
  words=[];
  leng;
  keyword=[];
  contextes=[];
  fileUrl;
  constructor(private downloaddata:UserService,private sanitizer: DomSanitizer,private toastr: ToastrService) { }

  ngOnInit(): void {

  let token= localStorage.getItem('token');
  
  const helper = new JwtHelperService();
  
  const decodedToken = helper.decodeToken(token);
  this.id=decodedToken.user_id;
  console.log(this.id)
  this.downloadrecent();
  
  /*const data = 'text: '+'\n' +'context :'+ '\n'+'keywords :';
  const blob = new Blob([data], { type: 'application/octet-stream' });
  this.fileUrl = this.sanitizer.bypassSecurityTrustResourceUrl(window.URL.createObjectURL(blob));*/

  }


  downloadrecent(){
    
  this.downloaddata.downloadrecent(this.id).subscribe(
    (result) => {
      this.res=result;
      console.log(this.res);
      var json = JSON.parse(JSON.stringify(result));
      //console.log(result['length'])
      this.leng=result['length']
      /*if (leng>3){
        leng=3;
      }*/

      for (let i = 0; i <this.leng; i++) {
      
        this.keys(result[i]['topic_id'],i);
              
      }
      
      
      //console.log(result[0]['topic_id']);
      /*this.resultat={
        date:json['document']['Date'],
        texte:json['document']['Text'],
        etiquette : json['contexte']['etiquette'],
        id : json['document']['topic_id'],
      }*/
      //console.log(this.resultat.id);
      
      //this.keys(this.resultat.id);
      
      
    },
    error => {
      console.log(error);
    }
  );
  
  }

  keys(pk,i){
    
    this.downloaddata.downloadkeys(pk).subscribe(
      result => {
        //console.log(result['contexte']);
        //this.keyss.push(result);
        this.keyword[i]=result['keys'];
        this.contextes[i]=result['contexte'];
        //console.log(this.keyss);
        //setTimeout(() => { console.log("World!"); }, 200); 
        //var json = JSON.parse(JSON.stringify(this.keyss[i].keys));

        //console.log(i)
        /*this.words[i] = {
          keyword0: json[0]['mots'],
          keyword1: json[1]['mots'],
          keyword2: json[2]['mots'],
          keyword3: json[3]['mots'],
          keyword4: json[4]['mots'],
          keyword5: json[5]['mots'],
          keyword6: json[6]['mots'],
          keyword7: json[7]['mots'],
          keyword8: json[8]['mots'],
          keyword9: json[9]['mots'],
          
         };*/
         //console.log(this.words);
         
         //console.log(this.keyword[i].keys[0]['mots']);

      },
      error=>{
        console.log(error);
      } 
    ); 

  }


  public downloadPDF():void {
    let DATA = this.htmlData.nativeElement;
    let doc = new jsPDF('l','pt', 'a4');
    doc.setFontSize(22);
    doc.setTextColor(0, 0, 0);
    doc.text(200, 30, 'Here are Your results of context-extraction!');
    let handleElement = {
      '#editor':function(element,renderer){
        return true;
      }
    };
    doc.setFont("helvetica");
    doc.setFontType("bold");
    doc.setFontSize(9);
    /*doc.fromHTML(DATA.innerHTML,15,15,{
      'width': 1200,
      'elementHandlers': handleElement
    });*/
    doc.setLineWidth(2);
    doc.autoTable({ html: '#table' })
    /*doc.autoTable({
      head: [['Name', 'Email', 'Country']],
      body: [
        ['David', 'david@example.com', 'Sweden'],
        ['Castille', 'castille@example.com', 'Spain'],
        // ...
      ],
    })*/
    //doc.setFontSize(22);
    //doc.setTextColor(0, 0, 0);
    //doc.text(200, 30, 'EASYContext-website');
    doc.save('EASYContext-Results.pdf');
    this.toastr.success('Your Results Are downloaded Successfully !');/*success/error/warning/info/show()*/
  }
  
}
